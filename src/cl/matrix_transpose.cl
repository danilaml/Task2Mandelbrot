//#define TILE_SIZE 32
#define TILE_SIZE 16

__kernel void matrix_transpose(__global const float *a, __global float *at, unsigned m, unsigned k)
{
    const size_t i = get_global_id(0);
    const size_t j = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];
    const size_t local_i = get_local_id(0);
    const size_t local_j = get_local_id(1);

    tile[j * TILE_SIZE][i] = a[j * k + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    // TODO: barriers
    at[i * m + j] = tile[j * TILE_SIZE][i];
}