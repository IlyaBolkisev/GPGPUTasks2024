#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(__global const float *A, __global const float *B, __global float *C, const unsigned int M, const unsigned int K, const unsigned int N)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    float sum = 0.0f;
    for (int k = 0; k < K; ++k)
        sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(__global const float *A, __global const float *B, __global float *C, const unsigned int M, const unsigned int K, const unsigned int N)
{
    __local float Asub[TILE_SIZE][TILE_SIZE];
    __local float Bsub[TILE_SIZE][TILE_SIZE];

    int global_row = get_global_id(0);
    int global_col = get_global_id(1);

    int local_row = get_local_id(0);
    int local_col = get_local_id(1);

    float sum = 0.0f;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        Asub[local_row][local_col] = A[global_row * K + t * TILE_SIZE + local_col];
        Bsub[local_row][local_col] = B[t * TILE_SIZE * N + local_row * N + global_col];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += Asub[local_row][k] * Bsub[k][local_col];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[global_row * N + global_col] = sum;
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(__global float *A, __global float *B, __global float *C, const unsigned int M, const unsigned int K, const unsigned int N)
{
    __local float Asub[TILE_SIZE][TILE_SIZE];
    __local float Bsub[TILE_SIZE][TILE_SIZE];

    int global_row = get_global_id(0);
    int global_col = get_global_id(1);

    int local_row = get_local_id(0);
    int local_col = get_local_id(1);

    float sum[WORK_PER_THREAD];
    for (int w = 0; w < WORK_PER_THREAD; w++)
        sum[w] = 0.0f;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++)
    {
        for (int w = 0; w < WORK_PER_THREAD; w++) {
            Asub[local_col * WORK_PER_THREAD + w][local_row] = A[(global_col * WORK_PER_THREAD + w) * K + t * TILE_SIZE + local_row];
            Bsub[local_col * WORK_PER_THREAD + w][local_row] = B[(t * TILE_SIZE + local_col * WORK_PER_THREAD + w) * N + global_row];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++)
            for (int w = 0; w < WORK_PER_THREAD; w++)
                sum[w] += Asub[local_col * WORK_PER_THREAD + w][k] * Bsub[k][local_row];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < WORK_PER_THREAD; w++)
        C[(global_col * WORK_PER_THREAD + w) * N + global_row] = sum[w];
}
#endif
