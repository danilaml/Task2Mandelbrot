#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/bitonic_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cfloat>

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

static unsigned next_pow2(unsigned v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

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

    as.resize(next_pow2(n), FLT_MAX);
    gpu::gpu_mem_32f as_gpu;
    as_gpu.resizeN(next_pow2(n));

    {
        ocl::Kernel bitonic(bitonic_kernel, bitonic_kernel_length, "bitonic");
        ocl::Kernel bitonic_global(bitonic_kernel, bitonic_kernel_length, "bitonic_global");
        ocl::Kernel bitonic_global_fst(bitonic_kernel, bitonic_kernel_length, "bitonic_global_fst");
        bitonic.compile();
        bitonic_global.compile();
        bitonic_global_fst.compile();

        const auto new_n = next_pow2(n);
        unsigned int workGroupSize = 256;
        unsigned int global_work_size = (new_n + workGroupSize - 1) / workGroupSize * workGroupSize / 2;
        auto work_size = gpu::WorkSize(workGroupSize, global_work_size);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), new_n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            bitonic.exec(work_size, as_gpu, new_n);
            for (unsigned k = workGroupSize * 4; k <= new_n; k *= 2) {
                bitonic_global_fst.exec(work_size, as_gpu, new_n, k);
                for (unsigned st = k / 4; st >= 1; st /= 2) {
                    bitonic_global.exec(work_size, as_gpu, new_n, st);
                }
            }

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (new_n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
