#define WORK_GROUP_SIZE 256

__kernel void sum(__global const unsigned *xs, unsigned n, __global unsigned *res)
{
    const size_t localId = get_local_id(0);
    const size_t globalId = get_global_id(0);

    __local unsigned local_xs[WORK_GROUP_SIZE];
    if (globalId >= n)
        return;
    local_xs[localId] = xs[globalId];

    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned nvalues = WORK_GROUP_SIZE; nvalues > 1; nvalues /= 2) {
        if (2 * localId < nvalues) {
            unsigned a = local_xs[localId];
            unsigned b = local_xs[localId + nvalues/2];
            local_xs[localId] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (localId == 0) {
        atomic_add(res, local_xs[0]);
    }
}