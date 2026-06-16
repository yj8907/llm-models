

import triton


def get_cuda_autotune_config():

    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
    ]

@triton.autotune(
        configs=get_cuda_autotune_config(),
        keys=['M','N','K']
)
@triton.jit
def matmul_triton_kernel(a_ptr, b_ptr, c_ptr, M, N, K, 
                   stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
                   BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                   GROUP_SIZE_M: tl.constexpr):
    
    pid = tl.program_id(axis=0)

    num_block_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)

    group_idx = pid // (GROUP_SIZE_M*num_block_n)
    m_idx_within_group = (pid - group_idx * GROUP_SIZE_M * num_block_n) % GROUP_SIZE_M
    n_idx_within_group = (pid - group_idx * GROUP_SIZE_M * num_block_n) // GROUP_SIZE_M

    block_m_idx = group_idx*GROUP_SIZE_M + m_idx_within_group
    block_n_idx = n_idx_within_group

    acc_c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N))

    offs_m = (tl.arange(0, BLOCK_SIZE_M)*stride_am)[:, None] + \
              (tl.arange(0, BLOCK_SIZE_K)*stride_ak)[None, :]
    
    offs_n = (tl.arange(0, BLOCK_SIZE_N)*stride_bn)[:, None] + \
                (tl.arange(0, BLOCK_SIZE_K)*stride_bk)[None, :]

    a_ptr += block_m_idx*stride_am
    b_ptr += block_n_idx*stride_bn
    for _ in range(0, K, BLOCK_SIZE_K):
        a_block_mk = tl.load(a_ptr+offs_m)
        b_block_nk = tl.load(b_ptr+offs_n)

        acc_c += tl.dot(a_block_mk, tl.trans(b_block_nk, (1, 0))) # BLOCK_SIZE_M, BLOCK_SIZE_N

        a_ptr += BLOCK_SIZE_K*stride_ak
        b_ptr += BLOCK_SIZE_K*stride_bk

    c_ptr += block_m_idx*stride_cm + block_n_idx*stride_cn
    offs_c = (tl.arange(0, BLOCK_SIZE_M)*stride_cm)[:, None] + \
              (tl.arange(0, BLOCK_SIZE_N)*stride_cn)[None, :]
    
    tl.store(c_ptr+offs_c, acc_c)

    return

def ref_kernel():

    grid = lambda META: tl.cdiv(M, META['BLOCK_SIZE_M'])*tl.cdiv(N, META['BLOCK_SIZE_N'])

    matmul_triton_kernel[grid]()

    return