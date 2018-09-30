#define WORK_GROUP_SIZE 256

__kernel void calc_prefs(__global int *xs
                        , const int n
                        , __global int *sums
                        , const unsigned is_not_last)
{
    const size_t localId = get_local_id(0);
    const size_t globalId = get_global_id(0);
    const unsigned max_sz = min(n, WORK_GROUP_SIZE * 2);
    
    unsigned offset = 1;
    
    __local int local_xs[WORK_GROUP_SIZE * 2];
    local_xs[2 * localId] = (2 * globalId >= n) ? 0 : xs[2 * globalId];
    local_xs[2 * localId + 1] = (2 * globalId + 1 >= n) ? 0 : xs[2 * globalId + 1];
    int last_x = 0;
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
            int x = local_xs[ai];
            local_xs[ai] = local_xs[bi];
            local_xs[bi] += x;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    xs[2 * globalId] = local_xs[2 * localId];
    xs[2 * globalId + 1] = local_xs[2 * localId + 1];
    if (localId == 0 && is_not_last) {
        sums[get_group_id(0)] = local_xs[WORK_GROUP_SIZE * 2 - 1] + last_x;
    }
}


__kernel void add_sums(__global int *xs, __global const int *sums)
{
    const size_t groupId = get_group_id(0);
    const size_t globalId = get_global_id(0);

    int s = sums[groupId];
    xs[globalId * 2] += s;
    xs[globalId * 2 + 1] += s;
}

__kernel void find_max(const __global int *xs, const int n, __global int *max_res, const int last)
{
    const size_t localId = get_local_id(0);
    const size_t globalId = get_global_id(0);

    __local int local_xs[WORK_GROUP_SIZE];
    local_xs[localId] = xs[globalId];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (unsigned nvalues = WORK_GROUP_SIZE; nvalues > 1; nvalues /= 2) {
        if (2 * localId < nvalues && (localId + nvalues/2 < n)) {
            int a = local_xs[localId];
            int b = local_xs[localId + nvalues/2];
            
            local_xs[localId] = max(a, b);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_xs[0] > *max_res) {
        atomic_max(max_res, local_xs[0]);
    }
    if (globalId + 1 == n) {
        atomic_max(max_res, last + xs[globalId]);
    }
}

__kernel void find_index(const __global int *xs, const int value, __global int *index)
{
    const size_t globalId = get_global_id(0);
    
    if (xs[globalId] == value) {
        atomic_min(index, globalId);
    }
}
