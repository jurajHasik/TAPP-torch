# TAPP extension for PyTorch

[TAPP interface](https://github.com/TAPPorg/tensor-interfaces) as (composable) PyTorch operator extension via [Stable ABI](https://docs.pytorch.org/cppdocs/stable.html#torch-stable-api)

For general binary tensor contraction plus addition $D= \alpha AB + \beta C$

````python
tapp_torch.ops.tensor_product(A: Tensor, B: Tensor, C: Union[Tensor,None], D: Tensor, 
    modes_A: Sequence[int], modes_B: Sequence[int], modes_C: Union[Sequence[int],None], modes_D: Sequence[int], 
    alpha: Union[float,complex,Tensor,None], beta: Union[float,complex,Tensor,None]) -> None:
````

and a tensordot subset with autograd and torch.compile support

````python
tapp_torch.ops.tensordot(A: Tensor, B: Tensor, 
    contracted_modes_A: List[int], contracted_modes_B: List[int],
    modes_out: Optional[List[int]]=None) -> Tensor:
````

Requires Pytorch 2.10+

## Installation

Get dependencies  `TAPP`, [`TBLIS`](https://github.com/MatthewsResearchGroup/tblis) as git submodules. These are cloned to `third-party/`.

``` 
git submodule update --init --recursive
```
Get [`cuTensor`](https://developer.nvidia.com/cutensor), i.e. via `pip`, see [`cutensor-cu<XX>`](https://pypi.org/search/?q=cutensor).

Next, build TAPP with (optionally) `TBLIS` and (optionally) `cuTensor`, here with CMake.

NOTE: In case of builds within `conda`, you might need to specify i.e. `CC=cc CXX=g++` to get compilers recognized by CMake.

NOTE: If no TBLIS_SOURCE_DIR is provided, TAPP's CMake checkouts TBLIS 


```bash
mkdir third-party/tapp/build && cd third-party/tapp/build
cmake -DTAPP_REFERENCE_BUILD_CUTENSOR_BINDS=ON -DTAPP_REFERENCE_ENABLE_TBLIS=ON \
    -DTAPP_REFERENCE_TBLIS_SOURCE_DIR=../../tblis ..
make -j <number-of-cores>
```

Finally, build and install PyTorch extension (from the root of the repo)

NOTE: If needed, adjust the path to TAPP build in `setup.py` accordingly. By default it is set to `third_party/tapp/build`.

NOTE: set USE_CUDA=0 to build cpu-only extension.

```
pip install --no-build-isolation -e .
```

## Testing

To run the tests for the custom operators:

First, get optional deps, here from `pyproject.toml`

```bash
pip install --no-build-isolation -e ".[tests]"
```

For TAPP's general `tensor_product`

```bash
pytest tests/test_tapp_torch.py::TestTensorProduct -s
```

For `tensordot` subset, including gradients and torch.compile

```bash
pytest tests/test_tapp_torch.py::TestTensordot -s
```

## Performance benchmarks

See `benchmarks` folder.

### Requirements

TAPP, TBLIS, cuTensor 2.5+, Pytorch 2.10+


### References

https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0

https://github.com/pytorch/extension-cpp

https://docs.pytorch.org/cppdocs/stable.html#torch-stable-api

https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func
