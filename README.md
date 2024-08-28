# pycuda-example

A small working CUDA example powered by Python and 
[PyCUDA](https://pypi.org/project/pycuda/). 
Go to [main.py](main.py) to view the code.

The example implements an algorithm that counts the relatively non-prime numbers
(GCD is not 1) for each number up to `size=1000`.

## Requirements

* NVIDIA device
* NVIGIA driver installed (https://www.nvidia.com/en-us/drivers/)
* CUDA installed (https://developer.nvidia.com/cuda-downloads)
* Python 3.9 (or higher)
* numpy>=2.0.2
* pycuda>=2024.1.2

## How to run

1. Create a virtualenv: `python -m virtualenv .venv`
2. Activate the virtualenv: `source ./.venv/bin/activate` (or `.\.venv\Scripts\activate` for Windows)
3. Install requirements: `pip install -r requirements.txt`
4. Run the script: `python main.py`
