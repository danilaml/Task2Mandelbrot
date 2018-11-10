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
        if (offset * k + k - 1 - k_i < WG_SIZE * 2) {
            const float a = local_xs[offset * k + k_i];
            const float b = local_xs[offset * k + k - 1 - k_i];
            if (a > b) {
                local_xs[offset * k + k_i] = b;
                local_xs[offset * k + k - 1 - k_i] = a;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned st = k / 4; st >= 1; st /= 2) {
            const unsigned offset_st = local_i / st;
            const unsigned st_i = local_i % st;
            if (offset_st * 2 * st + st_i + st < WG_SIZE * 2) {
                const float a = local_xs[offset_st * 2 * st + st_i];
                const float b = local_xs[offset_st * 2 * st + st_i + st];
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
               unsigned aCount, 
               unsigned bCount, 
               int diag)
{
    int beg = max(0, diag - (int)bCount);
    int end = min(diag, (int)aCount);

    while (beg < end) {
        int mid = (beg + end) / 2;
        if (a[mid] <= b[diag - 1 - mid]) {
            beg = mid + 1;
        } else {
            end = mid;
        }
    }
    return beg;
}

__kernel void merge(const __global float *as, __global float *res, unsigned count) {
    const unsigned group_id = get_group_id(0);
    const unsigned local_id = get_local_id(0);
    const unsigned offset = group_id * count * 2;
    const unsigned items_per_wi = 2 * count / get_local_size(0);
    const int diag = local_id * items_per_wi;
    const __global float *a = as + offset;
    const __global float *b = as + offset + count;
    const int mp = merge_path(a, b, count, count, diag);
    
    unsigned curA = mp;
    unsigned curB = diag - mp;
    float aValue = a[curA];
    float bValue = b[curB];
    
    #pragma unroll 4
    for (unsigned i = 0; i < items_per_wi; ++i) {
        if ((curB >= count) || ((curA < count) && aValue <= bValue)) {
            res[offset + diag + i] = aValue;
            aValue = a[++curA];
        } else {
            res[offset + diag + i] = bValue;
            bValue = b[++curB];
        }
    }
}

__kernel void merge_mp(const __global float *as, const __global int *mps, __global float *res, unsigned n, unsigned count) {
    const unsigned group_id = get_group_id(0);
    const unsigned local_id = get_local_id(0);

    const unsigned array_id = (group_id * 256 * 8) / (2 * count);
    const unsigned gl_diag = (group_id * 256 * 8) % (2 * count); 

    const unsigned gl_offset = array_id * count * 2;

    const __global float *gl_a = as + gl_offset;
    const __global float *gl_b = as + gl_offset + count;

    const int gl_mp = mps[group_id];
    const unsigned a0 = gl_mp;
    const unsigned b0 = gl_diag - gl_mp;
    const int gl_mp_nxt = ((group_id + 1) % (count / 1024) == 0) ? count : mps[group_id + 1];
    const unsigned aCount = gl_mp_nxt - a0;
    const unsigned bCount = (gl_diag + 2048 - gl_mp_nxt) - b0;

    const unsigned items_per_wi = 8;
    const int diag = local_id * items_per_wi;
    const __global float *a = gl_a + a0;
    const __global float *b = gl_b + b0;
    const int mp = merge_path(a, b, aCount, bCount, diag);
    
    unsigned curA = mp;
    unsigned curB = diag - mp;
    float aValue = a[curA];
    float bValue = b[curB];
    
    #pragma unroll 8
    for (unsigned i = 0; i < items_per_wi; ++i) {
        if ((curB >= bCount) || ((curA < aCount) && aValue <= bValue)) {
            res[gl_offset + gl_diag + diag + i] = aValue;
            aValue = a[++curA];
        } else {
            res[gl_offset + gl_diag + diag + i] = bValue;
            bValue = b[++curB];
        }
    }
}

__kernel void get_merge_path(const __global float *as, __global int *mps, unsigned n, unsigned count) {
    const unsigned global_id = get_global_id(0);
    const unsigned ids_per_array = count / 1024;
    const unsigned array_id = global_id / ids_per_array;
    const unsigned id_in_array = global_id % ids_per_array; 
    const int diag = id_in_array * 2 * 1024;

    const unsigned offset = array_id * count * 2;

    if (offset + count >= n)
        return;

    const __global float *a = as + offset;
    const __global float *b = as + offset + count;

    const int mp = merge_path(a, b, count, count, diag);
    mps[global_id] = mp;
}
