"""
Benchmarks for tensor network contractions using cotengra + tapp_torch.

Usage
-----
    python benchmarks/bench_contraction.py [OPTIONS]

The actual tensor network to contract is defined by ``build_network()``.
Replace its body with the desired contraction once the network is decided.
"""
import argparse
import time
from typing import Optional

import torch
import cotengra as ctg

from benchmarks.backend import BACKEND_TAPP, BACKEND_TORCH, setup_backends
from benchmarks.networks.ctm_1x1 import TN_CTM_1X1
from benchmarks.networks.ctm_2x3 import TN_CTM_2X3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}

NETWORKS: dict[str, callable] = {
    "ctm_1x1": TN_CTM_1X1.build_network,
    "ctm_2x3": TN_CTM_2X3.build_network,
}

def build_network(name: str= None, **kwargs) -> tuple:
    """
    Build and return the tensor network to benchmark.

    Returns
    -------
    name: str
        Optional name of a predefined network to build. If None, a default example is returned.
    inputs : list[tuple[str, ...]]
        Index labels for each tensor (cotengra / opt_einsum format).
    output : tuple[str, ...]
        Index labels for the output tensor.
    size_dict : dict[str, int]
        Mapping from index label to dimension size.

    Default example :
        A_{i,j,k}  B_{j,k,l}  C_{l,m} -> A_{i,m}
    """
    if name is not None and name not in NETWORKS:
        raise ValueError(
            f"Unknown network {name!r}. "
            f"Available: {', '.join(NETWORKS.keys())}"
        )
    if name is not None:
        return NETWORKS[name](**kwargs)
    
    inputs = [
        ("i", "j", "k"),
        ("j", "k", "l"),
        ("l", "m"),
    ]
    output = ("i", "m")
    size_dict = {"i": 64, "j": 64, "k": 64, "l": 64, "m": 64}
    
    return inputs, output, size_dict


def find_contraction_path(
    inputs,
    output,
    size_dict,
    *,
    optimize: str = "auto",
    max_repeats: int = 128,
):
    """
    Use cotengra to find a (near-)optimal contraction tree.

    Parameters
    ----------
    optimize : str
        Optimizer name passed to ``cotengra.HyperOptimizer``.
    max_repeats : int
        Number of random trials for the hyper-optimizer.

    Returns
    -------
    tree : cotengra.ContractionTree
    """
    opt = ctg.HyperOptimizer(
        methods=[optimize] if optimize != "auto" else None,
        max_repeats=max_repeats,
        minimize='flops', max_time=None, parallel=False, 
        simulated_annealing_opts=None, slicing_opts=None, slicing_reconf_opts=None, 
        reconf_opts=None, optlib=None, space=None, score_compression=0.75, 
        on_trial_error='warn', max_training_steps=None, progbar=False, )
    tree= ctg.array_contract_tree(inputs, output, size_dict, shapes=None, 
        optimize=opt, canonicalize=True, sort_contraction_indices=False)
    return tree


def make_random_tensors(inputs, size_dict, *, dtype, device):
    """Create random tensors matching the network specification."""
    arrays = []
    for ix in inputs:
        shape = tuple(size_dict[i] for i in ix)
        arrays.append(torch.randn(shape, dtype=dtype, device=device))
    return arrays


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def _warmup(arrays):
    """Warm up by doing a simple in-place operation on all tensors."""
    for arr in arrays:
        arr.mul_(1.0)  # in-place to avoid extra allocations


def _warpup_tapp(dtype,device):
    """Warm up tapp_torch by running a small dummy tensordot."""
    inputs, output, size_dict= build_network()
    tree = find_contraction_path(
        inputs, output, size_dict,
        optimize="auto",
        max_repeats=128,
    )
    arrays = make_random_tensors(inputs, size_dict, dtype=dtype, device=device)
    tree.contract(arrays, order=None, prefer_einsum=False, strip_exponent=False, 
            check_zero=False, backend=BACKEND_TAPP, implementation=None, autojit=False, progbar=False)


def bench_torch(tree, arrays, *, n_runs: int = 10):
    """Benchmark using plain PyTorch (torch.tensordot) via cotengra."""

    def run():
        return tree.contract(arrays, order=None, prefer_einsum=False, strip_exponent=False, 
            check_zero=False, backend=BACKEND_TORCH, implementation=None, autojit=False, progbar=False)

    times = []
    for _ in range(n_runs):
        _warmup(arrays)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def bench_tapp(tree, arrays, *, n_runs: int = 10):
    """Benchmark using tapp_torch backend via cotengra + autoray."""

    # Wrap tensors so autoray recognises the backend name.
    # autoray dispatches based on the module that owns the type;
    # since our tensors are plain torch.Tensors we tell cotengra
    # to use our backend explicitly.
    def run():
        return tree.contract(arrays, order=None, prefer_einsum=False, strip_exponent=False, 
            check_zero=False, backend=BACKEND_TAPP, implementation=None, autojit=False, progbar=False)

    # invoke once to trigger (perhaps?) torch.ops registration
    print("TAPP_torch loaded")

    times = []
    for _ in range(n_runs):
        _warmup(arrays)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(label: str, times: list[float]):
    import statistics

    med = statistics.median(times)
    best = min(times)
    worst = max(times)
    print(f"  {label:>12s}:  median {med*1e3:8.2f} ms | "
          f"best {best*1e3:8.2f} ms | worst {worst*1e3:8.2f} ms  "
          f"({len(times)} runs)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_network_kwarg(s: str) -> tuple[str, object]:
    """Parse a ``key=value`` string, attempting int/float conversion."""
    key, _, val = s.partition("=")
    if not val:
        raise argparse.ArgumentTypeError(
            f"Network parameter must be key=value, got: {s!r}"
        )
    # Try int, then float, then keep as string
    for convert in (int, float):
        try:
            return key, convert(val)
        except ValueError:
            continue
    return key, val


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark tensor network contraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available networks: {NETWORKS.keys()}\n\n"
               "Network parameters are passed as --net-param key=value, e.g.:\n"
               "  --network ctm_2x3 --net-param D=4 --net-param X=64",
    )
    p.add_argument(
        "--network", type=str, default=None,
        help=f"Tensor network to benchmark (default: None). "
             f"Choices: {NETWORKS.keys()}",
    )
    p.add_argument(
        "--net-param", type=_parse_network_kwarg, action="append",
        default=[], dest="net_params", metavar="KEY=VALUE",
        help="Parameter for the network factory (repeatable)",
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        help="Device to run on (default: cpu)",
    )
    p.add_argument(
        "--dtype", type=str, default="float64",
        choices=list(DTYPE_MAP.keys()),
        help="Data type (default: float64)",
    )
    p.add_argument(
        "--n-runs", type=int, default=20,
        help="Number of timed iterations (default: 20)",
    )
    p.add_argument(
        "--optimizer", type=str, default="auto",
        help="cotengra optimizer method (default: auto)",
    )
    p.add_argument(
        "--max-repeats", type=int, default=128,
        help="cotengra hyper-optimizer repeats (default: 128)",
    )
    p.add_argument(
        "--skip-torch", action="store_true",
        help="Skip the plain-torch baseline",
    )
    p.add_argument(
        "--skip-tapp", action="store_true",
        help="Skip the tapp_torch benchmark",
    )
    p.add_argument(
        "--verify", action="store_true",
        help="Crosscheck results between torch and tapp_torch (with a single run, not timed)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    dtype = DTYPE_MAP[args.dtype]
    device = args.device
    net_kwargs = dict(args.net_params)

    # 1. Register the tapp_torch and torch_default autoray backend
    setup_backends()

    # 2. Build network
    inputs, output, size_dict = build_network(args.network, **net_kwargs)
    eq = ",".join("".join(ix if isinstance(ix,str) else f"'{ix}'") for ix in inputs) + "->" + "".join(output)
    print(f"  equation: {eq}")
    print(f"  tensors:  {len(inputs)}")
    print(f"  dtype={args.dtype}  device={device}")
    print()

    # 3. Find contraction path
    print("Finding contraction path …")
    tree = find_contraction_path(
        inputs, output, size_dict,
        optimize=args.optimizer,
        max_repeats=args.max_repeats,
    )
    print(tree.contract_stats())
    tree.print_contractions()
    print()

    # 4. Create tensors
    arrays = make_random_tensors(inputs, size_dict, dtype=dtype, device=device)

    # 5. Benchmark
    print(f"Benchmarking ({args.n_runs} runs each) …")

    if not args.skip_torch:
        times_torch = bench_torch(tree, arrays, n_runs=args.n_runs)
        report("torch", times_torch)

    if not args.skip_tapp:
        _warpup_tapp(dtype,device)
        times_tapp = bench_tapp(tree, arrays, n_runs=args.n_runs)
        report("tapp_torch", times_tapp)

    # 6. Verify correctness (optional quick check)
    if args.verify:
        print()
        print("Verifying correctness …")
        ref = tree.contract(arrays, backend="torch")
        tapp_out = tree.contract(arrays, backend=BACKEND_TAPP)
        if torch.allclose(ref, tapp_out, atol=1e-5, rtol=1e-5):
            print(f"torch {ref.norm()} TAPP_torch {tapp_out.norm()} Results match.")
        else:
            max_diff = (ref - tapp_out).abs().max().item()
            print(f"torch {ref.norm()} TAPP_torch {tapp_out.norm()} Max absolute diff = {max_diff}")


if __name__ == "__main__":
    main()