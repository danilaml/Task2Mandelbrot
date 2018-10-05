#define TILE_SIZE 16

__kernel void matrix_multiplication(__global const float *a, __global const float *b, __global float *c, unsigned M, unsigned K, unsigned N)
{
    const size_t i = get_global_id(0);
    const size_t j = get_global_id(1);
    const size_t local_i = get_local_id(0);
    const size_t local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    for (unsigned tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        barrier(CLK_LOCAL_MEM_FENCE);
        tileA[local_j][local_i] = a[j * K + tileK * TILE_SIZE + local_i]; // k
        tileB[local_j][local_i] = b[(local_j + tileK * TILE_SIZE) * N + i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[local_j][k] * tileB[k][local_i];
        }
    }
    c[j * N + i] = sum;
}