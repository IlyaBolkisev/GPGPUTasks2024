#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6
#define TILE_SIZE 16

__kernel void matrix_transpose_naive(__global float *A, __global float *B, const unsigned int M, const unsigned int K)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < M && col < K) {
        B[col * M + row] = A[row * K + col];
    }
}


__kernel void matrix_transpose_local_bad_banks(__global float *A, __global float *B, const unsigned int M, const unsigned int K)
{
    int global_row = get_global_id(0);
    int global_col = get_global_id(1);

    int local_row = get_local_id(0);
    int local_col = get_local_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];

    int tile_row = get_group_id(0);
    int tile_col = get_group_id(1);

    tile[local_col][local_row] = A[global_col * M + global_row];
    barrier(CLK_LOCAL_MEM_FENCE);

    B[(tile_row * TILE_SIZE + local_col) * K + tile_col * TILE_SIZE + local_row] = tile[local_row][local_col];
}


__kernel void matrix_transpose_local_good_banks(__global float *A, __global float *B, const unsigned int M, const unsigned int K)
{
    int global_row = get_global_id(0);
    int global_col = get_global_id(1);

    int local_row = get_local_id(0);
    int local_col = get_local_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE+1];

    int tile_row = get_group_id(0);
    int tile_col = get_group_id(1);

    tile[local_col][local_row] = A[global_col * M + global_row];
    barrier(CLK_LOCAL_MEM_FENCE);

    B[(tile_row * TILE_SIZE + local_col) * K + tile_col * TILE_SIZE + local_row] = tile[local_row][local_col];
}
