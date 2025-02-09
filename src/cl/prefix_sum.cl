__kernel void prefix_sum(__global unsigned int* data, __global unsigned int* res, const unsigned int bias, const unsigned int n)
{
    int gid = get_global_id(0);
    if (bias <= gid) {
        res[gid] = data[gid - bias] + data[gid];
    }
    else
    {
        res[gid] = data[gid];
    }
}