# TAPP extension for PyTorch

[TAPP interface](https://github.com/TAPPorg/tensor-interfaces) as (composable) PyTorch operator extension via Stable ABI

```
tapp_torch.ops.tensor_product(A: Tensor, B: Tensor, C: Union[Tensor,None], D: Tensor, 
                   modes_A: Sequence[int], modes_B: Sequence[int], modes_C: Union[Sequence[int],None], modes_D: Sequence[int],
                   alpha: Union[float,complex,Tensor,None], beta: Union[float,complex,Tensor,None]) -> None:
```

Requires Pytorch 2.10+

## Installation

Get dependencies  `TAPP`, [`TBLIS`](https://github.com/MatthewsResearchGroup/tblis) as git submodules. These are cloned to `third-party/`.

``` 
git submodule update --init --recursive
```

Build TAPP with TBLIS, here with CMake.

NOTE: In case of builds within `conda`, specify i.e. `CC=cc CXX=g++` to get compilers recognized by CMake

NOTE: If no TBLIS_SOURCE_DIR is provided, TAPP's CMake checkouts TBLIS 


```
mkdir third-party/tapp/build && cd third-party/tapp/build
cmake -DTAPP_REFERENCE_ENABLE_TBLIS=ON -DTAPP_REFERENCE_TBLIS_SOURCE_DIR=../../tblis ..
make -j <number-of-cores>
```

Build and install PyTorch extension (from the root of the repo)

```
pip install --no-build-isolation -e .
```

## Testing

To run the tests for the custom operators:

For TAPP's general `tensor_product`

```bash
pytest tests/test_tapp_torch.py::TestTensorProduct -s
```

For tensordot subset, including gradients and torch.compile

```bash
pytest tests/test_tapp_torch.py::TestTensordot -s
```


This will run the unit tests defined in `test_custom_op.py` to ensure the correctness of the operator's functionality.


## Requirements

TAPP, TBLIS, Pytorch 2.10+