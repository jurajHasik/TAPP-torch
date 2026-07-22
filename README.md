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

The `v2` variants are faster, encoding block-sparse structure data (`*_blocks`, `*_strides`, `*_offsets`) directly as int64 CPU `Tensor`s instead of `Sequence[int]`, avoiding per-element Python→C++ boxing overhead for tensors with large numbers of blocks.

````python
tapp_torch.ops.tensor_product_bs_v2(A: Tensor, B: Tensor, C: Union[Tensor,None], D: Tensor,
        a_modes: Sequence[int], a_numSectionsPerMode: Sequence[int], a_sectionExtents: Sequence[int],
        a_blocks: Tensor, a_strides: Tensor, a_offsets: Tensor,
        b_modes: Sequence[int], b_numSectionsPerMode: Sequence[int], b_sectionExtents: Sequence[int],
        b_blocks: Tensor, b_strides: Tensor, b_offsets: Tensor,
        c_modes: Union[Sequence[int],None], c_numSectionsPerMode: Union[Sequence[int],None], c_sectionExtents: Union[Sequence[int],None],
        c_blocks: Union[Tensor,None], c_strides: Union[Tensor,None], c_offsets: Union[Tensor,None],
        d_modes: Sequence[int], d_numSectionsPerMode: Sequence[int], d_sectionExtents: Sequence[int],
        d_blocks: Tensor, d_strides: Tensor, d_offsets: Tensor,
        alpha: Union[float,complex,Tensor,None], beta: Union[float,complex,Tensor,None],
        descriptor_key_hashes: Optional[Sequence[int]]=None) -> None:
````

````python
tapp_torch.ops.tensordot_bs_v2(A: Tensor, B: Tensor,
        contracted_modes_A: Sequence[int], contracted_modes_B: Sequence[int],
        a_numSectionsPerMode: Sequence[int], a_sectionExtents: Sequence[int],
        a_blocks: Tensor, a_strides: Tensor, a_offsets: Tensor,
        b_numSectionsPerMode: Sequence[int], b_sectionExtents: Sequence[int],
        b_blocks: Tensor, b_strides: Tensor, b_offsets: Tensor,
        d_numSectionsPerMode: Sequence[int], d_sectionExtents: Sequence[int],
        d_blocks: Tensor, d_strides: Tensor, d_offsets: Tensor,
        modes_out: Optional[Sequence[int]]=None,
        descriptor_key_hashes: Optional[Sequence[int]]=None) -> Tensor:
````

Both `v2` ops additionally accept an optional `descriptor_key_hashes`: a flat sequence of 24 ints
— 3 consecutive 512-bit (8×int64) digests, one each for the A, B, D block-sparse descriptor keys
(`numSectionsPerMode + sectionExtents + blocks + strides`). If the caller already has a stable,
unique identifier for each block-sparse structure (e.g. computed once and cached on the caller
side), passing it here skips the O(number of blocks) hashing the descriptor/contraction-plan
caches would otherwise perform on every call. **The caller is responsible for the digest actually
being unique to its structure** — it is trusted as the sole cache identity, with no fallback
comparison against the underlying arrays. See `TAPP_CACHE_STRICT` below to sanity-check this.

Requires Pytorch 2.10+

## Examples

For dense tensor contractions see `examples/tensordot_example.py`.

For block-sparse tensor contractions see `examples/tensordot_bs_example.py`.
Light-weight interface for block-sparse tensors is provided by [symmray](https://github.com/jcmgray/symmray).

## Performance benchmarks

See `benchmarks` folder.

## Caching

Control caching of cuTensor block-sparse descriptors and plans. 

* Block-sparse tensor descriptor cache: Size can be set via ``TAPP_DESCRIPTOR_CACHE_SIZE`` environment variable.
Set to 0 for unlimited cache or to 4 or higher. Default is 1024.

* Block-sparse contraction descriptor and plan cache: Size can be set via ``TAPP_PLAN_CACHE_SIZE`` environment variable.
Set to 0 for unlimited cache or to 1 or higher. Default is 256.

* Set ``TAPP_CACHE_STRICT=1`` to ignore any caller-supplied `descriptor_key_hashes` (see the `v2`
ops above) and always use the strictly-correct, full-array-comparison cache path. Useful for
verifying that a caller's hash computation is actually correct: run once normally and once with
``TAPP_CACHE_STRICT=1`` and confirm identical results. Default is 0 (off).

## Additional settings

* Set ``CUTENSOR_BLOCKSPARSE_REPRODUCIBLE=1`` for bit-wise reproducibility. Default is 0.
* Control logging by setting ``TAPP_LOG_LEVEL=<1..6>`` with 6 for highest verbosity level.

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
* `CUTENSOR_ROOT=<path-to-cutensor-root-directory>` to use different than default cuTensor installation

Alternatively, go to `third_party/tapp` and build TAPP directly

```bash
mkdir third-party/tapp/build && cd third-party/tapp/build
cmake -DCMAKE_BUILD_TYPE=Release -DTAPP_CUTENSOR_BINDINGS=ON -DTAPP_REFERENCE_USE_TBLIS=ON \
    -DTAPP_REFERENCE_TBLIS_SOURCE_DIR=../../tblis ..
cmake -j <number-of-cores>
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
pytest tests/test_tapp_bs_torch.py::TestTensorProductBs -s
```

For `tensordot` subset, including gradients but **without torch.compile**

```bash
pytest tests/test_tapp_bs_torch.py::TestTensordotBs -s
```

### Requirements

TAPP, Pytorch 2.10+, cuTensor 2.5+, (optional) TBLIS 


### References

https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0

https://github.com/pytorch/extension-cpp

https://docs.pytorch.org/cppdocs/stable.html#torch-stable-api

https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func
