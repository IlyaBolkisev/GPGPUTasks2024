#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif
#line 6

__kernel void sum_atomic(__global const unsigned int *arr, __global unsigned int *sum, unsigned int n) {
    const unsigned int gid = get_global_id(0);
    if (gid >= n)
        return;
    atomic_add(sum, arr[gid]);
}


__kernel void sum_loop(__global const unsigned int* arr, __global unsigned int* sum, unsigned int n)
{
    unsigned int gid = get_global_id(0);
    unsigned int gsize = get_global_size(0);

    unsigned int partial = 0;
    for (unsigned int i = gid; i < n; i += gsize)
        partial += arr[i];

    atomic_add(sum, partial);
}


__kernel void sum_loop_coalesced(__global const unsigned int* arr, __global unsigned int* sum, unsigned int n)
{
    unsigned int gid = get_global_id(0);
    unsigned int gsize = get_global_size(0);

    unsigned int partial = 0;
    for (unsigned int i = gid; i < n; i += gsize)
        partial += arr[i];

    atomic_add(sum, partial);
}


__kernel void sum_local(__global const unsigned int* arr, __global unsigned int* sum, unsigned int n)
{
    __local unsigned int shared_data[256];

    unsigned int lid = get_local_id(0);
    unsigned int lsize = get_local_size(0);

    unsigned int gid = get_global_id(0);
    unsigned int gsize = get_global_size(0);

    unsigned int partial = 0;
    for (unsigned int i = gid; i < n; i += gsize)
        partial += arr[i];

    shared_data[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int offset = lsize / 2; offset > 0; offset >>= 1) {
        if (lid < offset)
            shared_data[lid] += shared_data[lid + offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add(sum, shared_data[0]);
    }
}


__kernel void sum_tree(__global const unsigned int* arr, __global unsigned int* sum, unsigned int n)
{
    __local unsigned int local_data[256];

    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);
    unsigned int group_size = get_local_size(0);
    unsigned int global_size = get_global_size(0);

    unsigned int partial = 0;
    for (unsigned int i = gid; i < n; i += global_size)
        partial += arr[i];

    local_data[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int offset = group_size >> 1; offset > 0; offset >>= 1) {
        if (lid < offset)
            local_data[lid] += local_data[lid + offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add(sum, local_data[0]);
    }
}
