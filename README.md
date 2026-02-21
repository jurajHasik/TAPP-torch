# TAPP extension for PyTorch

## Dense tensors

[TAPP interface](https://github.com/TAPPorg/tensor-interfaces) as (composable) PyTorch operator extension via [Stable ABI](https://docs.pytorch.org/cppdocs/stable.html#torch-stable-api)

For general binary **dense** tensor contraction plus addition $D= \alpha AB + \beta C$

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

## Block-sparse tensors

For general binary block-sparse tensor contraction plus addition $D= \alpha AB + \beta C$

````python
tapp_torch.ops.tensor_product_bs(A: Tensor, B: Tensor, C: Union[Tensor,None], D: Tensor, 
        a_modes: Sequence[int], a_numSectionsPerMode: Sequence[int], a_sectionExtents: Sequence[int], 
        a_blocks: Sequence[int], a_strides:  Sequence[int], a_offsets: Sequence[int],
        b_modes: Sequence[int], b_numSectionsPerMode: Sequence[int], b_sectionExtents: Sequence[int], 
        b_blocks: Sequence[int], b_strides:  Sequence[int], b_offsets: Sequence[int],
        c_modes: Union[Sequence[int],None], c_numSectionsPerMode: Union[Sequence[int],None], c_sectionExtents: Union[Sequence[int],None], 
        c_blocks: Union[Sequence[int],None], c_strides: Union[Sequence[int],None], c_offsets: Union[Sequence[int],None],
        d_modes: Sequence[int], d_numSectionsPerMode:  Sequence[int], d_sectionExtents: Sequence[int], 
        d_blocks: Sequence[int], d_strides:  Sequence[int], d_offsets: Sequence[int],
        alpha: Union[float,complex,Tensor,None], beta: Union[float,complex,Tensor,None]) -> None:
````

and a tensordot subset with autograd. **torch.compile is currently not supported.**

````python
tapp_torch.ops.tensordot_bs(A: Tensor, B: Tensor,
        contracted_modes_A: Sequence[int], contracted_modes_B: Sequence[int],
        a_numSectionsPerMode: Sequence[int], a_sectionExtents: Sequence[int], 
        a_blocks: Sequence[int], a_strides:  Sequence[int], a_offsets: Sequence[int],
        b_numSectionsPerMode: Sequence[int], b_sectionExtents: Sequence[int], 
        b_blocks: Sequence[int], b_strides:  Sequence[int], b_offsets: Sequence[int],
        d_numSectionsPerMode:  Sequence[int], d_sectionExtents: Sequence[int], 
        d_blocks: Sequence[int], d_strides:  Sequence[int], d_offsets: Sequence[int], 
        modes_out: Optional[Sequence[int]]=None) -> Tensor:
````

Requires Pytorch 2.10+

## Examples

For dense tensor contractions see `examples/tensordot_example.py`.

For block-sparse tensor contractions see `examples/tensordot_bs_example.py`.
Light-weight interface for block-sparse tensors is provided by [symmray](https://github.com/jcmgray/symmray).

## Performance benchmarks

See `benchmarks` folder.

## Installation

### TL;DR

Get dependencies  `TAPP`, [`TBLIS`](https://github.com/MatthewsResearchGroup/tblis) as git submodules. These are cloned to `third-party/`. 

``` 
git submodule update --init --recursive
```
Get [`cuTensor`](https://developer.nvidia.com/cutensor), i.e. via `pip`, see [`cutensor-cu<XX>`](https://pypi.org/search/?q=cutensor).
Then build TAPP and `tapp_torch` extension

```
pip install --no-build-isolation -e .
```

### In depth

**First**, `pip` automatically builds TAPP via CMake inside `third_party/tapp/build` with the process being controlled by the following environment variables

* `TAPP_CUTENSOR_BINDINGS=ON` to build cuTensor bindings
* `TAPP_REFERENCE_USE_TBLIS=ON` to enable TBLIS for dense tensor contractions on `cpu`
* `TAPP_REFERENCE_TBLIS_SOURCE_DIR=<path-to-tblis-source>` provide custom TBLIS source directory. Default set to `../../tblis`, which is to location of `tblis` submodule with respect to default build directory,
* `TAPP_FORCE_BUILD=1` to rebuild TAPP i.e. clean default build dir and re-run CMake

Alternatively, go to `third_party/tapp` and build TAPP directly

```bash
mkdir third-party/tapp/build && cd third-party/tapp/build
cmake -DTAPP_CUTENSOR_BINDINGS=ON -DTAPP_REFERENCE_USE_TBLIS=ON \
    -DTAPP_REFERENCE_TBLIS_SOURCE_DIR=../../tblis ..
make -j <number-of-cores>
```

NOTE: *In case of builds within `conda`, you might need to specify i.e. `CC=cc CXX=g++` to get compilers recognized by CMake.*

NOTE: *If no TBLIS_SOURCE_DIR is provided, TAPP's CMake checkouts TBLIS*

**Second**, build and install `tapp_torch` PyTorch extension (from the root of the repo)

```
pip install --no-build-isolation -e .
```

the pre-built TAPP is detected and `pip` proceeds to directly build the extension (provided `TAPP_FORCE_BUILD` is not set).

NOTE: *set `USE_CUDA=0` to build cpu-only extension.*

### Troubleshooting

Run verbose install 

```
pip install -v --no-build-isolation -e .
```

and look for `TAPP_torch` reporting on building TAPP in addition to other issues.

## Testing

To run the tests for the custom operators:

First, get optional deps, here from `pyproject.toml`

```bash
pip install --no-build-isolation -e ".[tests]"
```

### Dense tensors

For TAPP's general `tensor_product`

```bash
pytest tests/test_tapp_torch.py::TestTensorProduct -s
```

For `tensordot` subset, including gradients and torch.compile

```bash
pytest tests/test_tapp_torch.py::TestTensordot -s
```

### Block-sparse tensors

```bash
pytest tests/test_tapp_torch.py::TestTensorProductBs -s
```

For `tensordot` subset, including gradients but **without torch.compile**

```bash
pytest tests/test_tapp_torch.py::TestTensordotBs -s
```

### Requirements

TAPP, Pytorch 2.10+, cuTensor 2.5+, (optional) TBLIS 


### References

https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0

https://github.com/pytorch/extension-cpp

https://docs.pytorch.org/cppdocs/stable.html#torch-stable-api

https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func
