#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"

#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

void runTest(const std::string &kernel_name, std::vector<unsigned int> &as, unsigned int ref_sum)
{
    gpu::gpu_mem_32u as_gpu;
    unsigned int n = as.size();
    unsigned int work_group_size = 256;
    unsigned int global_work_size = n + work_group_size - 1;
    const int benchmarkingIters = 100;

    as_gpu.resizeN(n);
    as_gpu.writeN(as.data(), n);

    ocl::Kernel sum_gpu_kernel(sum_kernel, sum_kernel_length, kernel_name);
    sum_gpu_kernel.compile();

    timer t;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
        unsigned int sum = 0;

        gpu::gpu_mem_32u sum_gpu;
        sum_gpu.resizeN(1);
        sum_gpu.writeN(&sum, 1);
        sum_gpu_kernel.exec(gpu::WorkSize(work_group_size, global_work_size), as_gpu, sum_gpu, n);
        sum_gpu.readN(&sum, 1);

        EXPECT_THE_SAME(ref_sum, sum, "GPU " + kernel_name + " result should be consistent!");

        t.nextLap();
    }

    std::cout << "[" << kernel_name << "]" << std::endl;
    std::cout << "    GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "    GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
}

int main(int argc, char **argv)
{

    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        runTest("sum_atomic", as, reference_sum);          // 3.2.2
        runTest("sum_loop", as, reference_sum);            // 3.2.2
        runTest("sum_loop_coalesced", as, reference_sum);  // 3.2.3
        runTest("sum_local", as, reference_sum);           // 3.2.4
        runTest("sum_tree", as, reference_sum);
    }
}
