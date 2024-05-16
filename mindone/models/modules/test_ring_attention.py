import utils
import mindspore as ms
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import Tensor, nn, ops
from ring_attention import RingAttention
from mindspore.ops.operations.nn_ops import FlashAttentionScore

def generate_inputs(B, N1, N2, S1, S2, D, input_layout, dtype, return_tensor=True):
    min_value = -1
    max_value = 1
    np.random.seed(42)
    if input_layout == "BSH":
        query = np.random.uniform(min_value, max_value, [B, S1, N1 * D])
        key = np.random.uniform(min_value, max_value, [B, S2, N2 * D])
        value = np.random.uniform(min_value, max_value, [B, S2, N2 * D])
    elif input_layout == "BNSD":
        query = np.random.uniform(min_value, max_value, [B, N1, S1, D])
        key = np.random.uniform(min_value, max_value, [B, N2, S2, D])
        value = np.random.uniform(min_value, max_value, [B, N2, S2, D])
    elif input_layout == "SBH":
        query = np.random.uniform(min_value, max_value, [S1, B, N1 * D])
        key = np.random.uniform(min_value, max_value, [S2, B, N2 * D])
        value = np.random.uniform(min_value, max_value, [S2, B, N2 * D])
    elif input_layout == "BSND":
        query = np.random.uniform(min_value, max_value, [B, S1, N1, D])
        key = np.random.uniform(min_value, max_value, [B, S2, N2, D])
        value = np.random.uniform(min_value, max_value, [B, S2, N2, D])
    elif input_layout == "TND":
        query = np.random.uniform(min_value, max_value, [B * S1, N1, D])
        key = np.random.uniform(min_value, max_value, [B * S2, N2, D])
        value = np.random.uniform(min_value, max_value, [B * S2, N2, D])
    else:
        raise ValueError(f"input_layout is invalid.")
    real_shift = None
    prefix = None
    drop_mask = None
    attn_mask = None
    padding_mask = None
    if return_tensor:
        return Tensor(query, dtype=dtype), Tensor(key, dtype=dtype), Tensor(value, dtype=dtype), real_shift, drop_mask, padding_mask, attn_mask, prefix
    return query, key, value, real_shift, drop_mask, padding_mask, attn_mask, prefix

if __name__ == '__main__':
    # If the device_target is GPU, set the device_target to "GPU"
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

    # Init parameter
    sp = 8
    utils.init_sp_group(sp)

    dtype = mstype.float16
    B1 = 1
    N1 = 8
    S1 = 4096
    D1 = 16
    head_num1 = N1
    query, key, value, real_shift, drop_mask, padding_mask, attn_mask, prefix = generate_inputs(B1, N1, N1, S1, S1, D1, "SBH", dtype)

    q2 = utils.get_sp_chuncks(query)
    k2 = utils.get_sp_chuncks(key)
    v2 = utils.get_sp_chuncks(value)

    ring_attention = RingAttention(head_num=head_num1, pre_tokens=k2.shape[0], next_tokens=0, keep_prob=1., input_layout="SBH")
    ring_attention_output = ring_attention(q2, k2, v2, real_shift, drop_mask, padding_mask, attn_mask)
    attn_mask = ops.ones((query.shape[0], key.shape[0]), dtype=ms.uint8)
    attn_mask = ops.triu(attn_mask, diagonal=1)
    flash_attention = FlashAttentionScore(head_num=head_num1, input_layout="SBH")
    _, _, _, flash_attention_output = flash_attention(query, key, value, real_shift, drop_mask, padding_mask, attn_mask)

    cur_rank = utils.get_rank()
    sp_group_size = utils.get_sequence_parallel_world_size()
    chunck_num = sp_group_size * 2
    chunck_size = S1/chunck_num

    flash_attention_output = utils.get_sp_chuncks(flash_attention_output, seq_dim = 0)

    assert np.allclose(flash_attention_output.asnumpy(), ring_attention_output.asnumpy(), 0.004, 0.004)

    print("end test.")
