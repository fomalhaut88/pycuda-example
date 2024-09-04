#include <stdio.h>


typedef unsigned int uint;


// Euclidean algorithm to calculate GCD
__device__ uint calc_gcd(uint a, uint b) {
    while (b > 0) {
        uint t = b;
        b = a % b;
        a = t;
    }
    return a;
}


// Kernel function
__global__ void kernel(uint size, const uint* nums, uint* result) {
    uint i = threadIdx.x + blockIdx.x * blockDim.x;
    uint j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < size) {
        if (j < i) {
            uint a = nums[i];
            uint b = nums[j];

            if ((a < 2) || (b < 2)) {
                return;
            }

            if (calc_gcd(a, b) != 1) {
                atomicAdd(result + i, 1);
            }
        }
    }
}


int main() {
    // Define the size
    const uint SIZE = 1000;

    // Define the input array as number from `0` to `size - 1`
    uint nums[SIZE];
    for (uint i = 0; i < SIZE; i++) {
        nums[i] = i;
    }

    // Copy input array to GPU
    uint* nums_gpu;
    cudaMalloc(&nums_gpu, SIZE * sizeof(uint));
    cudaMemcpy(nums_gpu, nums, SIZE * sizeof(uint), cudaMemcpyHostToDevice);

    // Define the result array with zeros
    uint* result_gpu;
    cudaMalloc(&result_gpu, SIZE * sizeof(uint));
    cudaMemset(result_gpu, 0, SIZE * sizeof(uint));

    // Basic call
    dim3 grid(SIZE, (SIZE + 1023) / 1024);
    dim3 block(1, 1024);
    kernel<<<grid, block>>>(SIZE, nums_gpu, result_gpu);

    // Copy the result from GPU
    uint result[SIZE];
    cudaMemcpy(result, result_gpu, SIZE * sizeof(uint), cudaMemcpyDeviceToHost);

    // Print the result array
    for (uint i = 0; i < SIZE; i++) {
        printf("%u ", result[i]);
    }
    printf("\n");

    // Free GPU memory
    cudaFree(nums_gpu);
    cudaFree(result_gpu);

    // Return 0
    return 0;
}
