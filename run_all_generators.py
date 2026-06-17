"""Run every generated PBR texture generator with one shared parameter set.

The generators created by ``Create_Textures_Models_Code.py`` are stored as:

    all_dirs/<topic>/<generator_name>/generate.py

Each ``generate.py`` is expected to expose:

    generate_texture(outdir=<path>, sz=<int>, numsamples=<int>)

This script walks all generated model directories, skips outputs that already
have enough sample folders/files, and records failures so long runs can resume.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType


# Generation parameters.
TEXTURE_SIZE = 1024 #PBR size
NUM_SAMPLES = 10 # Max PBR to generate
MAX_ATTEMPTS = 6

# The generated model tree and run logs live outside this source directory.
BASE_DIR = Path("")
GENERATORS_DIR = BASE_DIR / "output_pbrs" # generated PBR dirs
ERROR_LOG_FILE = BASE_DIR / f"fail_runs_{TEXTURE_SIZE}.json"
ATTEMPT_LOG_FILE = BASE_DIR / f"attempt_file_{TEXTURE_SIZE}.json"

# Output names written inside each generator directory.
OUTPUT_SUBDIR = f"Samples_{TEXTURE_SIZE}_{NUM_SAMPLES}"
TIME_FILE = f"Time_{TEXTURE_SIZE}_{NUM_SAMPLES}.txt"
ERROR_FILE = f"error_{TEXTURE_SIZE}"


def load_json(path: Path) -> dict:
    """Load a JSON dictionary, returning an empty one for missing log files."""
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def save_json(path: Path, data: dict) -> None:
    """Write a JSON dictionary with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=4)


def count_existing_outputs(outdir: Path) -> int:
    """Count non-hidden entries in an output directory."""
    if not outdir.exists():
        return 0
    return sum(1 for entry in outdir.iterdir() if not entry.name.startswith("."))


def load_generator_module(code_dir: Path) -> ModuleType:
    """Import one generated ``generate.py`` file as an isolated module."""
    module_path = code_dir / "generate.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Missing generator file: {module_path}")

    # Use the full directory in the module name to avoid cache collisions when
    # many generated folders all contain a file called generate.py.
    safe_module_suffix = str(code_dir.resolve()).replace(os.sep, "_").replace(":", "_")
    module_name = f"generated_texture_{safe_module_suffix}"

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    added_to_path = str(code_dir) not in sys.path
    if added_to_path:
        sys.path.insert(0, str(code_dir))

    try:
        spec.loader.exec_module(module)
    finally:
        if added_to_path:
            sys.path.remove(str(code_dir))

    if not hasattr(module, "generate_texture"):
        raise AttributeError(f"{module_path} does not define generate_texture")
    return module


def run_generator(code_dir: Path, num_samples: int, size: int, outdir: Path) -> None:
    """Run one generated texture script and let exceptions reach the caller."""
    module = load_generator_module(code_dir)
    module.generate_texture(outdir=str(outdir), sz=size, numsamples=num_samples)


def log_failure(
    errors: dict,
    key: str,
    code_dir: Path,
    error: Exception | str,
    size: int,
    num_samples: int,
) -> None:
    """Record a generator failure in the global JSON log and local error file."""
    message = str(error)
    errors[key] = {"error": message, "sz": size, "num_samples": num_samples}
    print(errors[key])

    save_json(ERROR_LOG_FILE, errors)
    with (code_dir / ERROR_FILE).open("w", encoding="utf-8") as file_obj:
        file_obj.write(f"{message}\n")


def iter_generator_dirs(root_dir: Path):
    """Yield ``(topic_name, generator_dir)`` pairs from the generated tree."""
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Generator root does not exist: {root_dir}")

    for topic_dir in sorted(root_dir.iterdir()):
        if not topic_dir.is_dir():
            continue

        for generator_dir in sorted(topic_dir.iterdir()):
            if generator_dir.is_dir():
                yield topic_dir.name, generator_dir


def main() -> None:
    """Run pending generated texture scripts and update resume logs."""
    errors = load_json(ERROR_LOG_FILE)
    attempts = load_json(ATTEMPT_LOG_FILE)

    counter = 0
    for topic_name, code_dir in iter_generator_dirs(GENERATORS_DIR):
        counter += 1
        generator_name = code_dir.name
        attempt_key = f"{topic_name}/{generator_name}"
        out_dir = code_dir / OUTPUT_SUBDIR

        print(counter, ":", topic_name, ")", generator_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Skip generators that already produced the requested number of outputs.
        if count_existing_outputs(out_dir) >= NUM_SAMPLES:
            continue

        attempts[attempt_key] = attempts.get(attempt_key, 0) + 1
        save_json(ATTEMPT_LOG_FILE, attempts)

        # If repeated runs terminate the interpreter or machine, the attempt log
        # still lets the next run stop retrying this generator forever.
        if attempts[attempt_key] > MAX_ATTEMPTS:
            error_message = "No Python error message. The process may have crashed or run out of memory."
            log_failure(
                errors,
                attempt_key,
                code_dir,
                error_message,
                TEXTURE_SIZE,
                NUM_SAMPLES,
            )
            continue

        print(out_dir)
        start = time.perf_counter()
        try:
            run_generator(code_dir, NUM_SAMPLES, TEXTURE_SIZE, out_dir)
        except Exception as err:
            log_failure(errors, attempt_key, code_dir, err, TEXTURE_SIZE, NUM_SAMPLES)
            continue

        elapsed = time.perf_counter() - start
        print("Time in sec:", elapsed)
        with (code_dir / TIME_FILE).open("w", encoding="utf-8") as file_obj:
            file_obj.write(f"{elapsed}\nPer Sample:{elapsed / NUM_SAMPLES}\n")


if __name__ == "__main__":
    main()
