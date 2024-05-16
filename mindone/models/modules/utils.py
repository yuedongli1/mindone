from mindspore import Tensor
from mindspore.communication import init, get_rank, get_group_size, create_group

_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_GLOBAL_RANKS = None


def init_sp_group(
    sp: int = 1,
    ) -> None:
    """Initialize parallel groups."""
    #assert torch.distributed.is_initialized()
    init()

    world_size = get_group_size()

    dp = world_size // sp

    # Build the context-parallel groups.
    global _SEQUENCE_PARALLEL_GROUP
    global _SEQUENCE_PARALLEL_GLOBAL_RANKS
    # assert _SEQUENCE_PARALLEL_GROUP is None, 'sequence parallel group is already initialized'

    for j in range(dp):
        start_rank = j * sp
        end_rank = (j + 1) * sp
        ranks = list(range(start_rank, end_rank))
        sp_group_name = "sp_group"
        create_group(sp_group_name, ranks)
        _SEQUENCE_PARALLEL_GROUP = sp_group_name
        _SEQUENCE_PARALLEL_GLOBAL_RANKS = ranks


def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    assert _SEQUENCE_PARALLEL_GROUP is not None, 'sequence parallel group is not initialized'
    return _SEQUENCE_PARALLEL_GROUP


def get_sequence_parallel_global_ranks():
    """Get all global ranks of the sequence parallel group that the caller rank belongs to."""
    assert (
    _SEQUENCE_PARALLEL_GLOBAL_RANKS is not None
    ), 'sequence parallel group is not initialized'
    return _SEQUENCE_PARALLEL_GLOBAL_RANKS


def get_sequence_parallel_world_size():
    """Return world size for the sequence parallel group."""
    return get_group_size(group=get_sequence_parallel_group())


def get_sequence_parallel_rank():
    """Return my rank for the sequence parallel group."""
    return get_rank(group=get_sequence_parallel_group())


def get_sp_chuncks(batch, rank=None, seq_dim=0):
    """
    Slice batch input along sequence dimension into multiple chunks,
    which are parallelized across NPUs in a sequence parallel group.
    """
    cp_size = get_sequence_parallel_world_size()
    if cp_size > 1:
        if rank is None:
            cp_rank = get_sequence_parallel_rank()
        else:
            cp_rank = rank

        if seq_dim == 0:
            batch = batch.view(
                2 * cp_size,
                batch.shape[seq_dim] // (2 * cp_size),
                *batch.shape[(seq_dim + 1) :],
                )
        else:
            batch = batch.view(
                *batch.shape[0:seq_dim],
                2 * cp_size,
                batch.shape[seq_dim] // (2 * cp_size),
                *batch.shape[(seq_dim + 1) :],
                )

        index = Tensor([cp_rank, (2 * cp_size - cp_rank - 1)])
        batch = batch.index_select(seq_dim, index)

        if seq_dim == 0:
            batch = batch.view(-1, *batch.shape[(seq_dim + 2) :])
        else:
            batch = batch.view(*batch.shape[0:seq_dim], -1, *batch.shape[(seq_dim + 2) :])

    return batch

