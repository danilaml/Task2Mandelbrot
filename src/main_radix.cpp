#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

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

constexpr unsigned WORK_GROUP_SIZE = 256;

static void computePrefixes(const gpu::gpu_mem_32u &xs
    , int n
    , ocl::Kernel &calc_prefs
    , ocl::Kernel &add_sums) {
    if (n <= WORK_GROUP_SIZE * 2) {
        calc_prefs.exec(gpu::WorkSize(WORK_GROUP_SIZE, n / 2), xs, n, xs, 0);
    }
    else {
        const auto sums_size = n / 2 / WORK_GROUP_SIZE;
        auto sums = gpu::gpu_mem_32u::createN(sums_size);
        calc_prefs.exec(gpu::WorkSize(WORK_GROUP_SIZE, n / 2), xs, n, sums, 1);
        computePrefixes(sums, sums_size, calc_prefs, add_sums);
        add_sums.exec(gpu::WorkSize(WORK_GROUP_SIZE, n / 2), xs, sums);
    }

}

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
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
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

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    {
        ocl::Kernel radix_bits(radix_kernel, radix_kernel_length, "radix_bits");
        ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort");
        ocl::Kernel calc_prefs(radix_kernel, radix_kernel_length, "calc_prefs");
        ocl::Kernel add_sums(radix_kernel, radix_kernel_length, "add_sums");
        calc_prefs.compile();
        add_sums.compile();
        radix_bits.compile();
        radix_sort.compile();

        unsigned int workGroupSize = WORK_GROUP_SIZE;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        auto work_size = gpu::WorkSize(workGroupSize, global_work_size);
        
        const auto new_n = next_pow2(n);
        gpu::gpu_mem_32u bits_gpu;
        gpu::gpu_mem_32u sorted_gpu;
        bits_gpu.resizeN(new_n);
        sorted_gpu.resizeN(new_n);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            for (unsigned mask = 1; mask != 0; mask <<= 1) {
                radix_bits.exec(work_size, as_gpu, bits_gpu, n, mask);
                computePrefixes(bits_gpu, new_n, calc_prefs, add_sums);
                radix_sort.exec(work_size, as_gpu, sorted_gpu, bits_gpu, n, mask);
                as_gpu.swap(sorted_gpu);
            }

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        sorted_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
