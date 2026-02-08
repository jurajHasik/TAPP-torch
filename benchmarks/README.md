# Benchmarks

Benchmark tensor network contractions comparing default **PyTorch** (`torch.tensordot`)
against the **TAPP_torch** extension. 

Benchmarks use 
[cotengra](https://cotengra.readthedocs.io/) for contraction-path optimisation
and [autoray](https://autoray.readthedocs.io/) for backend dispatch.

## Requirements

Independently as

```bash
pip install cotengra autoray cmaes cotengrust cytoolz kahypar
```

or from top-level directory via `pyproject.toml`

```bash
pip install -e ".[benchmarks]" --no-build-isolation
```


Plus a working `tapp_torch` installation (see top-level README).

## Usage

```bash
# Default: cpu, float64
python -m benchmarks.bench_contraction

# On CUDA with float32
python -m benchmarks.bench_contraction --device cuda --dtype float32

# Choose one of predefined networks to contract and specify its extra parameters 
python -m benchmarks.bench_contraction --network ctm_2x3 --net-param D=6 --net-param X=64

# Skip plain-torch baseline
python -m benchmarks.bench_contraction --skip-torch

# Use a specific cotengra optimizer
python -m benchmarks.bench_contraction --optimizer greedy --max-repeats 64
```

## Structure

| File | Description |
|------|-------------|
| `backend.py` | Registers `tapp_torch` as an autoray backend (`tensordot`, `transpose`) |
| `bench_contraction.py` | CLI benchmark harness: path finding → tensor creation → timing [→ optional verification] |

## Defining the network

Edit `build_network()` in `bench_contraction.py` to return the desired
`(inputs, output, size_dict)` tuple describing the tensor network.
See `networks/...` for examples of predefined contractions.