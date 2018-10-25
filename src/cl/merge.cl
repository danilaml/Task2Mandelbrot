#define WG_SIZE 256

__kernel void bitonic_local(__global float *as, unsigned n)
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

int merge_path(const __global float *a,
                    const __global float *b, 
                    unsigned count, 
                    int diag)
{
    int beg = max(0, diag - (int)count);
    int end = min(diag, (int)count);

    while (beg < end) {
        int mid = (beg + end) / 2u;
        if (a[mid] <= b[diag - 1u - mid]) {
            beg = mid + 1u;
        } else {
            end = mid;
        }
    }
    return beg;
}

__kernel void merge(const __global float *as, __global float *res, unsigned n, unsigned count) {
    const unsigned group_id = get_group_id(0);
    const unsigned local_id = get_local_id(0);
    const unsigned offset = group_id * count * 2;
    const unsigned items_per_wi = 2 * count / get_local_size(0);
    const int diag = local_id * items_per_wi;
    const __global float *a = as + offset;
    const __global float *b = as + offset + count;
    const int mp = merge_path(a, b, count, diag/*+1*/);

    //printf("g_id: %d, offset: %d, diag: %d, ipw: %d, mp: %d\n", get_global_id(0), offset, diag, items_per_wi, mp);
    
    unsigned curA = mp;
    unsigned curB = diag - mp;
    float aValue = a[curA];
    float bValue = b[curB]; // check in range?
    //printf("g_id: %d, offset: %d\ndiag: %d, ipw: %d, mp: %d, a: %f, b: %f\n", get_global_id(0), offset, diag, items_per_wi, mp, aValue, bValue);
    for (unsigned i = 0; i < items_per_wi; ++i) {
        if ((curB >= count) || ((curA < count) && aValue <= bValue)) {
            res[offset + diag + i] = aValue;
            aValue = a[++curA];
        } else {
            res[offset + diag + i] = bValue;
            bValue = b[++curB];
        }
        //printf("%d\n", offset + diag + i);
    }
}