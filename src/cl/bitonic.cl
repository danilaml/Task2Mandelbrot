#define WG_SIZE 256

__kernel void bitonic(__global float *as, unsigned n)
{
    const size_t global_i = get_global_id(0);
    const size_t local_i = get_local_id(0);
    __local float local_xs[WG_SIZE * 2];
    
    local_xs[local_i * 2] = global_i * 2 < n ? as[global_i * 2] : FLT_MAX;
    local_xs[local_i * 2 + 1] = (global_i * 2 + 1) < n ? as[global_i * 2 + 1] : FLT_MAX;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (unsigned k = 2; k <= WG_SIZE * 2; k *= 2) {
        const unsigned offset = local_i * 2 / k;
        const unsigned k_i = local_i % (k / 2);
        if (offset * k + k - 1 - k_i < n) {
            float a = local_xs[offset * k + k_i];
            float b = local_xs[offset * k + k - 1 - k_i];
            if (a > b) {
                local_xs[offset * k + k_i] = b;
                local_xs[offset * k + k - 1 - k_i] = a;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned st = k / 4; st >= 1; st /= 2) {
            const unsigned offset_st = local_i / st;
            const unsigned st_i = local_i % st;
            if (offset_st * 2 * st + st_i + st < n && st_i < st) {
                float a = local_xs[offset_st * 2 * st + st_i];
                float b = local_xs[offset_st * 2 * st + st_i + st];
                if (a > b) {
                    local_xs[offset_st * 2 * st + st_i] = b;
                    local_xs[offset_st * 2 * st + st_i + st] = a;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (global_i * 2 <= n) {
        as[global_i * 2] = local_xs[local_i * 2];
        as[global_i * 2 + 1] = local_xs[local_i * 2 + 1];
    }
}


__kernel void bitonic_global_fst(__global float *as, unsigned n, unsigned k) {
    const size_t global_i = get_global_id(0);
    const unsigned offset = global_i * 2 / k;
    const unsigned k_i = global_i % (k / 2);
    if (offset * k + k - 1 - k_i < n) {
        float a = as[offset * k + k_i];
        float b = as[offset * k + k - 1 - k_i];
        if (a > b) {
            as[offset * k + k_i] = b;
            as[offset * k + k - 1 - k_i] = a;
        }
    }
}

__kernel void bitonic_global(__global float* as, unsigned n, unsigned st) {
    const size_t global_i = get_global_id(0);
    const unsigned offset_st = global_i / st;
    const unsigned st_i = global_i % st;

    if (offset_st * 2 * st + st_i + st < n) {
        float a = as[offset_st * 2 * st + st_i];
        float b = as[offset_st * 2 * st + st_i + st];
        if (a > b) {
            as[offset_st * 2 * st + st_i] = b;
            as[offset_st * 2 * st + st_i + st] = a;
        }
    }
}