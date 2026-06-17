"""Load and execute a generated texture module for pipeline validation."""

import os
import sys
import importlib.util


def run(code_dir, num_samples, sz, outdir):
    """Run ``generate.generate_texture`` from a generated code directory.

    The generated script is loaded from ``code_dir/generate.py`` under a unique
    module name so repeated tests do not reuse stale imports from previous
    generated texture scripts.
    """
    # Resolve the generated module path.
    code_dir = os.path.abspath(code_dir)
    module_path = os.path.join(code_dir, "generate.py")

    # Build a unique import name for this generated script.
    module_name = "generate_" + "".join(
        char if char.isalnum() else "_" for char in code_dir
    )

    # Temporarily allow local imports from the generated code directory.
    added = code_dir not in sys.path
    if added:
        sys.path.insert(0, code_dir)

    try:
        # Load generate.py directly from disk.
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load generated module: {module_path}")

        # Execute the module and call its required entrypoint.
        generate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generate)
        generate.generate_texture(outdir=outdir, sz=sz, numsamples=num_samples)
    finally:
        # Restore sys.path after the generated module finishes or crashes.
        if added:
            try:
                sys.path.remove(code_dir)
            except ValueError:
                pass

    return "Code executed successfully"
