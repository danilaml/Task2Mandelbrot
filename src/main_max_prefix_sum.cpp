#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/max_prefix_sum_cl.h"

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

static void computePrefixes(const gpu::gpu_mem_32i &xs
                    , int n
                    , ocl::Kernel &calc_prefs
                    , ocl::Kernel &add_sums) {
    if (n <= WORK_GROUP_SIZE * 2) {
        calc_prefs.exec(gpu::WorkSize(WORK_GROUP_SIZE, n / 2), xs, n, xs, 0);
    } else {
        const auto sums_size = n / 2 / WORK_GROUP_SIZE;
        auto sums = gpu::gpu_mem_32i::createN(sums_size);
        calc_prefs.exec(gpu::WorkSize(WORK_GROUP_SIZE, n / 2), xs, n, sums, 1);
        computePrefixes(sums, sums_size, calc_prefs, add_sums);
        add_sums.exec(gpu::WorkSize(WORK_GROUP_SIZE, n / 2), xs, sums);
    }
    
}

int main(int argc, char **argv)
{
    int benchmarkingIters = 10;
    int max_n = (1 << 24);
    
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    ocl::Kernel calc_prefs(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "calc_prefs");
    ocl::Kernel add_sums(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "add_sums");
    ocl::Kernel find_max(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "find_max");
    ocl::Kernel find_index(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "find_index");
    calc_prefs.compile();
    add_sums.compile();
    find_max.compile();
    find_index.compile();

    for (int n = 2; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = r.next(-values_range, values_range);
        }

        int reference_max_sum;
        int reference_result;
        {
            int max_sum = 0;
            int sum = 0;
            int result = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
                if (sum > max_sum) {
                    max_sum = sum;
                    result = i + 1;
                }
            }
            reference_max_sum = max_sum;
            reference_result = result;
        }
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")" << std::endl;

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = 0;
                int sum = 0;
                int result = 0;
                for (int i = 0; i < n; ++i) {
                    sum += as[i];
                    if (sum > max_sum) {
                        max_sum = sum;
                        result = i + 1;
                    }
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {

            unsigned int workGroupSize = WORK_GROUP_SIZE;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            auto work_size = gpu::WorkSize(workGroupSize, global_work_size);

            gpu::gpu_mem_32i as_gpu = gpu::gpu_mem_32i::createN(n);
            gpu::gpu_mem_32i max_sum_gpu = gpu::gpu_mem_32i::createN(1);
            gpu::gpu_mem_32i result_gpu = gpu::gpu_mem_32i::createN(1);

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = 0;
                int result = n;
                as_gpu.writeN(as.data(), as.size());
                max_sum_gpu.writeN(&max_sum, 1);
                result_gpu.writeN(&result, 1);

                computePrefixes(as_gpu, n, calc_prefs, add_sums);

                find_max.exec(work_size, as_gpu, n, max_sum_gpu, as.back());
                max_sum_gpu.readN(&max_sum, 1);
                find_index.exec(work_size, as_gpu, max_sum, result_gpu);
                
                max_sum_gpu.readN(&max_sum, 1);
                result_gpu.readN(&result, 1);

                EXPECT_THE_SAME(reference_max_sum, max_sum, "GPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
