import math

import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule


if __name__ == "__main__":
    # CUDA module source
    source = """
    typedef unsigned int uint;

    // Euclidean algorithm to calculate GCD
    __device__ uint calc_gcd(uint a, uint b) {
        if (a < b) {
            uint t = a;
            a = b;
            b = t;
        }

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
    """

    # Define the size
    size = 1000

    # Define the input array as number from `0` to `size - 1`
    nums = gpuarray.arange(0, size, 1, dtype=np.uint32)

    # Define the result array with zeros
    result = gpuarray.zeros(size, dtype=np.uint32)

    # Compile CUDA module and extract the function `kernel`
    module = SourceModule(source)
    kernel = module.get_function("kernel")

    # Basic call
    kernel(
        np.uint32(size), nums, result, 
        block=(1, 1024, 1), 
        grid=(size, math.ceil(size / 1024)),
    )

    # Alternative call (it is  faster, 
    # `prepare` should be called once before multiple `prepared_call`)
    #kernel.prepare("IPP")
    #kernel.prepared_call(
    #    (size, math.ceil(size / 1024)),
    #    (1, 1024, 1),
    #    size, nums.gpudata, result.gpudata,
    #)

    # Print the result array
    print(result.get())
