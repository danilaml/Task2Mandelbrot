#define TILE_SIZE 16

__kernel void matrix_transpose(__global const float *a, __global float *at, unsigned m, unsigned k)
{
    const size_t i = get_global_id(0);
    const size_t j = get_global_id(1);
    const size_t local_i = get_local_id(0);
    const size_t local_j = get_local_id(1);
    const size_t g_id_i = get_group_id(0);
    const size_t g_id_j = get_group_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];

    tile[local_j][local_i] = a[j * k + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    at[(g_id_i * TILE_SIZE + local_j) * k + (g_id_j * TILE_SIZE + local_i)] = tile[local_i][local_j];
}