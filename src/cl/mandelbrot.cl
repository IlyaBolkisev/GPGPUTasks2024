#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float* results,
                         int width, int height,
                         float fromX, float fromY,
                         float sizeX, float sizeY,
                         uint iters,
                         int smoothing)
{
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    if (gx >= width || gy >= height)
        return;

    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    float x0 = fromX + (gx + 0.5f) * (sizeX / width);
    float y0 = fromY + (gy + 0.5f) * (sizeY / height);

    float x = x0;
    float y = y0;
    int iter = 0;

    for (; iter < iters; ++iter) {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        if ((x * x + y * y) > threshold2)
            break;

    }

    float result = (float)iter;
    if (smoothing == 1 && iter != iters)
        result -= log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);

    result /= iters;
    results[gy * width + gx] = result;
}
