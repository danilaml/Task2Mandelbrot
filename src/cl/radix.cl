#define WORK_GROUP_SIZE 256u

__kernel void radix_bits(const __global unsigned *as, __global unsigned *bits, unsigned n, unsigned mask)
{
    const unsigned g_id = get_global_id(0);
    if (g_id >= n)
        return;
    bits[g_id] = (as[g_id] & mask) ? 0 : 1;
}

__kernel void radix_sort(const __global unsigned *as, __global unsigned *sorted, const __global unsigned *inds, __global unsigned *new_bits, unsigned n, unsigned mask)
{
    const unsigned g_id = get_global_id(0);
    if (g_id >= n)
        return;
    const unsigned a = as[g_id];
    const unsigned total = inds[n - 1] + ((as[n - 1] & mask) ? 0 : 1);
    unsigned ind = inds[g_id];
    if (a & mask) {
        ind = total + g_id - ind;
    }

    sorted[ind] = a;
    if (mask << 1)
        new_bits[ind] = (a & (mask << 1)) ? 0 : 1;
}

__kernel void calc_prefs(__global unsigned *xs
                        , const unsigned n
                        , __global unsigned *sums
                        , const unsigned is_not_last)
{
    const size_t localId = get_local_id(0);
    const size_t globalId = get_global_id(0);
    const unsigned max_sz = min(n, WORK_GROUP_SIZE * 2);
    
    unsigned offset = 1;
    
    __local unsigned local_xs[WORK_GROUP_SIZE * 2];
    local_xs[2 * localId] = (2 * globalId >= n) ? 0 : xs[2 * globalId];
    local_xs[2 * localId + 1] = (2 * globalId + 1 >= n) ? 0 : xs[2 * globalId + 1];
    unsigned last_x = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (localId == 0) {
        last_x = local_xs[WORK_GROUP_SIZE * 2 - 1];
    }

    for (unsigned t = max_sz >> 1; t > 0; t >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < t) {
            unsigned ai = offset * (2*localId + 1) - 1;
            unsigned bi = offset * (2*localId + 2) - 1;
            local_xs[bi] += local_xs[ai];
        }
        offset *= 2;
    }
    if (localId == 0) {
        local_xs[max_sz - 1] = 0;
    }
    for (unsigned t = 1; t < max_sz; t *= 2) {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (localId < t) {
            unsigned ai = offset * (2*localId + 1) - 1;
            unsigned bi = offset * (2*localId + 2) - 1;
            unsigned x = local_xs[ai];
            local_xs[ai] = local_xs[bi];
            local_xs[bi] += x;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (2 * globalId + 1 < n) {
        xs[2 * globalId] = local_xs[2 * localId];
        xs[2 * globalId + 1] = local_xs[2 * localId + 1];
    }
    if (localId == 0 && is_not_last) {
        sums[get_group_id(0)] = local_xs[WORK_GROUP_SIZE * 2 - 1] + last_x;
    }
}


__kernel void add_sums(__global unsigned *xs, __global const unsigned *sums)
{
    const size_t groupId = get_group_id(0);
    const size_t globalId = get_global_id(0);

    unsigned s = sums[groupId];
    xs[globalId * 2] += s;
    xs[globalId * 2 + 1] += s;
}