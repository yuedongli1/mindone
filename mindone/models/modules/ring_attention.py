from einops import rearrange
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Tensor, nn, ops
from mindspore.ops.operations._inner_ops import Send, Receive
from mindspore.ops.operations.nn_ops import FlashAttentionScore
from mindone.models.modules import utils


def flash_attn_p2p_communicate(rank, send_tensor, send_dst,
                                recv_tensor, recv_src,
                                cp_group, stream_send, stream_recv):
    """Point-to-point communications of KV and dKV in Attention with context parallelism"""
    send_recv_ops = []

    send_op = Send(0, send_dst, group=cp_group)
    send_op.add_prim_attr("dtype", mstype.float16)
    recv_op = Receive(0, recv_src, shape=recv_tensor.shape, dtype=recv_tensor.dtype, group=cp_group)

    if rank % 2 == 0:
        with ms.hal.StreamCtx(stream_send):
            send_op(send_tensor)
        with ms.hal.StreamCtx(stream_recv):
            recv_tensor = recv_op(Tensor(0.0, dtype=mstype.float16))
        send_recv_ops.append(stream_send)
        send_recv_ops.append(stream_recv)
    else:
        with ms.hal.StreamCtx(stream_recv):
            recv_tensor = recv_op(Tensor(0.0, dtype=mstype.float16))
        with ms.hal.StreamCtx(stream_send):
            send_op(send_tensor)

        send_recv_ops.append(stream_recv)
        send_recv_ops.append(stream_send)
    send_recv_reqs = send_recv_ops
    return send_recv_reqs, recv_tensor


def forward_update(prev_attn_out, prev_softmax_max, prev_softmax_sum,
                    cur_attn_out, cur_softmax_max, cur_softmax_sum):
    # update softmax_max
    softmax_max = ops.maximum(prev_softmax_max, cur_softmax_max)
    prev_scale = ops.exp(prev_softmax_max - softmax_max)
    cur_scale = ops.exp(cur_softmax_max - softmax_max)

    # update softmax_sum
    prev_softmax_sum_scaled = prev_softmax_sum * prev_scale
    cur_softmax_sum_scaled = cur_softmax_sum * cur_scale
    softmax_sum = prev_softmax_sum_scaled + cur_softmax_sum_scaled

    # out updating scale
    prev_out_scale = prev_softmax_sum_scaled / softmax_sum
    cur_out_scale = cur_softmax_sum_scaled / softmax_sum

    # [b, n, s, 8] -> [s, b, h]
    n = prev_out_scale.shape[1]
    h = prev_attn_out.shape[-1]
    d = h // n
    prev_out_scale = prev_out_scale[..., 0].unsqueeze(3).tile((1, 1, 1, d))
    prev_out_scale = rearrange(prev_out_scale.asnumpy(), 'b n s d -> s b (n d)')
    prev_out_scale = Tensor(prev_out_scale)

    cur_out_scale = cur_out_scale[..., 0].unsqueeze(3).tile((1, 1, 1, d))
    cur_out_scale = rearrange(cur_out_scale.asnumpy(), 'b n s d -> s b (n d)')
    cur_out_scale = Tensor(cur_out_scale)

    attn_out = prev_attn_out * prev_out_scale + cur_attn_out * cur_out_scale
    return attn_out, softmax_max, softmax_sum


class RingAttention(nn.Cell):
    """Attention implementation with context parallelism"""
    def __init__(self,
                head_num,
                keep_prob=1.0,
                scale_value=1.0,
                pre_tokens=2147483647,
                next_tokens=2147483647,
                input_layout="SBH", #当前只支持“SBH”
                sparse_mode=0
                ):
        super(RingAttention, self).__init__()
        self.head_num = head_num
        self.keep_prob = keep_prob
        self.scale_value = scale_value
        self.pre_tokens = pre_tokens
        self.next_tokens = next_tokens
        self.input_layout = input_layout
        self.sparse_mode = sparse_mode
        self.flash_attention = FlashAttentionScore(head_num = self.head_num,
                                                    keep_prob = self.keep_prob,
                                                    scale_value = self.scale_value,
                                                    pre_tokens = self.pre_tokens,
                                                    next_tokens = self.next_tokens,
                                                    input_layout = self.input_layout,
                                                    sparse_mode = self.sparse_mode)
        self.stream_send = ms.hal.Stream()
        self.stream_recv = ms.hal.Stream()

    def construct(self, q, k, v, real_shift=None, drop_mask=None, padding_mask=None, attn_mask=None):
        cp_group = utils.get_sequence_parallel_group()
        cp_size = utils.get_sequence_parallel_world_size()
        rank = utils.get_sequence_parallel_rank()
        cp_global_ranks = utils.get_sequence_parallel_global_ranks()
        send_dst = cp_global_ranks[(rank + 1) % cp_size]
        recv_src = cp_global_ranks[(rank + cp_size - 1) % cp_size]
        if attn_mask is None:
            attn_mask = ops.ones((q.shape[0], k.shape[0]), dtype=mstype.uint8)
            attn_mask = ops.triu(attn_mask, diagonal=1)

        # split chunk[i]~chunk[cp_size-i-1] into chunk[i] and chunk[cp_size-i-1],, [2s, b, h] -> [2, s, b, h]
        q, k, v = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in [q, k, v]]

        send_kv = ops.cat((k.unsqueeze(0), v.unsqueeze(0)), axis=0) # [2, 2, s, b, h]
        recv_tensor = None
        send_recv_ops = []
        attn_out, softmax_max, softmax_sum = None, None, None
        for i in range(cp_size):
            # wait until KV is received from recv_src
            if len(send_recv_ops) > 0:
                for send_recv_op in send_recv_ops:
                    send_recv_op.synchronize()
                send_kv = recv_tensor
            if i < cp_size - 1:
                recv_tensor = ms.numpy.empty_like(send_kv)
                send_recv_ops, recv_tensor = flash_attn_p2p_communicate(rank, send_kv, send_dst, recv_tensor,
                                                                        recv_src, cp_group, self.stream_send,
                                                                        self.stream_recv)
            if i == 0:
                cur_k, cur_v = k, v
            else:
                cur_k, cur_v = send_kv[0], send_kv[1] # [2, s, b, h]
            # if causal:
            cur_attn_mask = None
            if i == 0:
                # [2, s, b, h] -> [2s, b, h]
                cur_attn_mask = attn_mask
                cur_q, cur_k, cur_v = [x.view(-1, *x.shape[2:]) for x in [q, cur_k, cur_v]]
            elif i <= rank:
                # [2, s, b, h] -> [2s, b, h]
                cur_q = q.view(-1, *q.shape[2:])
                # only k[0] v[0] need to be calculated
                cur_k, cur_v = [x[0] for x in [cur_k, cur_v]]
            else:
                # only q[1] need to be calculated
                cur_q = q[1]
                # [2, s, b, h] -> [2s, b, h]
                cur_k, cur_v = [x.view(-1, *x.shape[2:]) for x in [cur_k, cur_v]]

            all_att_outs = self.flash_attention(cur_q,
                                                cur_k,
                                                cur_v,
                                                real_shift,
                                                drop_mask,
                                                padding_mask,
                                                cur_attn_mask)

            # if i <= rank: [2s, b, h], [b, n, 2s, 8], [b, n, 2s, 8]
            # else: [s, b, h], [b, n, s, 8], [b, n, s, 8]
            cur_attn_out = all_att_outs[3]
            cur_softmax_max = all_att_outs[0]
            cur_softmax_sum = all_att_outs[1]

            if i == 0:
                attn_out = cur_attn_out
                softmax_max = cur_softmax_max
                softmax_sum = cur_softmax_sum
            elif i <= rank:
                attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
                    attn_out, softmax_max, softmax_sum,
                    cur_attn_out, cur_softmax_max, cur_softmax_sum
                )
                attn_out, softmax_max, softmax_sum = attn_out_updated, softmax_max_updated, softmax_sum_updated
            else:
                # [2s, b, h] -> [2, s, b, h]
                attn_out = attn_out.view(2, attn_out.shape[0] // 2, *attn_out.shape[1:])
                # [b, n, 2s, 8] -> [b, n, 2, s, 8]
                softmax_max = softmax_max.view(softmax_max.shape[0], softmax_max.shape[1],
                                               2, softmax_max.shape[2] // 2, softmax_max.shape[-1])
                softmax_sum = softmax_sum.view(softmax_sum.shape[0], softmax_sum.shape[1],
                                               2, softmax_sum.shape[2] // 2, softmax_sum.shape[-1])
                attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
                    attn_out[1], softmax_max[:, :, 1, :, :], softmax_sum[:, :, 1, :, :],
                    cur_attn_out, cur_softmax_max, cur_softmax_sum
                )

                attn_out[1] = attn_out_updated.copy()
                softmax_max[:, :, 1, :, :] = softmax_max_updated.copy()
                softmax_sum[:, :, 1, :, :] = softmax_sum_updated.copy()
                # [2, s, b, h] -> [2s, b, h]
                attn_out = attn_out.view(-1, *attn_out.shape[2:])
                # [b, n, 2, s, 8] -> [b, n, 2s, 8]

                softmax_max = softmax_max.view(softmax_max.shape[0], softmax_max.shape[1], -1,
                                               softmax_max.shape[-1])
                softmax_sum = softmax_sum.view(softmax_sum.shape[0], softmax_sum.shape[1], -1,
                                               softmax_sum.shape[-1])

        return attn_out