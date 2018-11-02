#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/merge_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<float> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<float> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32f as_gpu;
    gpu::gpu_mem_32f res_gpu;
    gpu::gpu_mem_32i mp_gpu;
    as_gpu.resizeN(n);
    res_gpu.resizeN(n);
    mp_gpu.resizeN(n / 4);

    {
        ocl::Kernel bitonic_local(merge_kernel, merge_kernel_length, "bitonic_local");
        ocl::Kernel merge(merge_kernel, merge_kernel_length, "merge");
        ocl::Kernel merge_mp(merge_kernel, merge_kernel_length, "merge_mp");
        ocl::Kernel get_merge_path(merge_kernel, merge_kernel_length, "get_merge_path");
        bitonic_local.compile();
        merge.compile();

        const unsigned int workGroupSize = 256;
        const auto bitonic_work_size = gpu::WorkSize(workGroupSize, (n + workGroupSize - 1) / workGroupSize * workGroupSize);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных
            
            bitonic_local.exec(bitonic_work_size, as_gpu, n); // sorts 512-long chunks

            for (unsigned count = 512; count < n && count <= 4 * workGroupSize; count *= 2) {
                const auto work_size = gpu::WorkSize(workGroupSize, n / 2 / count * workGroupSize);
                merge.exec(work_size, as_gpu, res_gpu, count);
                as_gpu.swap(res_gpu);
            }

            for (unsigned count = 8 * workGroupSize; count < n; count *= 2) {
                const auto work_size_paths = gpu::WorkSize(workGroupSize, n / workGroupSize / 8);
                const auto work_size_merge = gpu::WorkSize(workGroupSize, n / 8);
                get_merge_path.exec(work_size_paths, as_gpu, mp_gpu, n, count);
                merge_mp.exec(work_size_merge, as_gpu, mp_gpu, res_gpu, n, count);
                as_gpu.swap(res_gpu);
            }

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}