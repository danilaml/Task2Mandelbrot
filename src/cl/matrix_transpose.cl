#define TILE_SIZE 16

__kernel void matrix_transpose(__global const float *a, __global float *at, unsigned m, unsigned k)
{
    const size_t i = get_global_id(0);
    const size_t j = get_global_id(1);

    at[i * m + j] = a[j * k + i];
}