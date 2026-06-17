"""
verify_pbr_output.py

Verifies the output of a PBR texture generation function.

Usage:
    error, message = verify_pbr_output(outdir, numsamples, sz)

Returns:
    (error: bool, error_message: str)
    error=False means all checks passed.
"""

import os
from typing import Optional
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_MAPS = {"BaseColor.png"}

OPTIONAL_MAPS = {
    "Roughness.png",
    "Metallic.png",
    "Height.png",
    "Transmission.png",
    "Emission.png",
    "Normal.png",
    "AmbientOcclusion.png",
    "Specular.png",
}

ALL_VALID_MAPS = REQUIRED_MAPS | OPTIONAL_MAPS

# Minimum value range (max - min) required in BaseColor for visible variation
BASECOLOR_MIN_RANGE = 20

# Minimum fraction of pixels that must differ between any two samples
# for a given map to be considered "different"
SAMPLE_DIFF_THRESHOLD = 0.01  # 1% of pixels must differ

# A map is considered "uniform" if the std-dev of its pixel values is below this
UNIFORMITY_STD_THRESHOLD = 1.0  # out of 0-255


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image_as_array(path: str) -> np.ndarray:
    """Load a PNG as a uint8 NumPy array shaped (H, W) or (H, W, C)."""
    img = Image.open(path)
    return np.array(img, dtype=np.uint8)


def _check_image_size(arr: np.ndarray, expected_sz: int, name: str) -> Optional[str]:
    """Return an error string if the image dimensions don't match expected_sz."""
    h, w = arr.shape[:2]
    if h != expected_sz or w != expected_sz:
        return f"'{name}': expected {expected_sz}x{expected_sz}, got {h}x{w}"
    return None


def _is_uniform(arr: np.ndarray) -> bool:
    """Return True if the image is effectively a solid colour (std < threshold)."""
    return float(arr.std()) < UNIFORMITY_STD_THRESHOLD




def _fraction_differing_pixels(a: np.ndarray, b: np.ndarray) -> float:
    """
    Fraction of pixel positions where at least one channel differs between a and b.

    Handles both grayscale (H, W) and multi-channel (H, W, C) arrays.
    Arrays must have the same spatial dimensions (verified upstream).
    """
    if a.ndim == 2 and b.ndim == 2:
        # Grayscale: direct per-pixel comparison
        differ = a != b
    else:
        # Promote grayscale to 3-D for uniform handling
        if a.ndim == 2:
            a = a[:, :, np.newaxis]
        if b.ndim == 2:
            b = b[:, :, np.newaxis]
        # Align channel counts (e.g. RGBA vs RGB) by trimming the larger one
        min_c = min(a.shape[2], b.shape[2])
        differ = np.any(a[:, :, :min_c] != b[:, :, :min_c], axis=2)

    return float(differ.mean())


# ---------------------------------------------------------------------------
# Main verification function
# ---------------------------------------------------------------------------

def verify_pbr_output(
    outdir: str,
    numsamples: int = 1,
    sz: int = 512,
):
    """
    Verify the output of a PBR texture generation function.

    Parameters
    ----------
    outdir      : str  - output directory passed to generate_texture()
    numsamples  : int  - number of samples that were requested
    sz          : int  - expected square resolution of each map

    Returns
    -------
    (error: bool, error_message: str)
        error=False  ->  all checks passed; error_message is "OK".
        error=True   ->  one or more checks failed; error_message lists them all.
    """
    errors = []

    # ------------------------------------------------------------------
    # 1. Output directory must exist
    # ------------------------------------------------------------------
    if not os.path.isdir(outdir):
        return True, f"Output directory does not exist: '{outdir}'"

    # ------------------------------------------------------------------
    # 2. Expected sample sub-folders must all be present
    # ------------------------------------------------------------------
    expected_folders = [f"sample_{i:03d}" for i in range(numsamples)]

    missing_folders = [
        f for f in expected_folders
        if not os.path.isdir(os.path.join(outdir, f))
    ]
    if missing_folders:
        errors.append(f"Missing sample folder(s): {missing_folders}")

    # Only process folders that actually exist
    present_folders = [
        f for f in expected_folders
        if os.path.isdir(os.path.join(outdir, f))
    ]

    # ------------------------------------------------------------------
    # 3. Per-sample checks
    # ------------------------------------------------------------------
    # Accumulate per-map arrays for cross-sample diversity comparison.
    # Lists are parallel to present_folders (same index = same sample).
    # structure: {map_name: [array_for_sample_0, array_for_sample_1, ...]}
    sample_maps = {}

    for folder in present_folders:
        folder_path = os.path.join(outdir, folder)

        # sorted() ensures deterministic iteration order
        png_files = sorted(
            f for f in os.listdir(folder_path) if f.lower().endswith(".png")
        )

        # ---- 3a. Required maps present --------------------------------
        for required in sorted(REQUIRED_MAPS):
            if required not in png_files:
                errors.append(f"[{folder}] Missing required map: '{required}'")

        # ---- 3b. At least 2 optional maps present ---------------------
        present_optional = [f for f in png_files if f in OPTIONAL_MAPS]
        if len(present_optional) < 2:
            errors.append(
                f"[{folder}] Only {len(present_optional)} optional map(s) found "
                f"(need at least 2). Present: {present_optional}"
            )

        # ---- 3c. Per-file: size, BaseColor diversity, and data collection ----
        for png_name in png_files:
            if png_name not in ALL_VALID_MAPS:
                continue  # already reported above; skip further checks

            png_path = os.path.join(folder_path, png_name)

            try:
                arr = _load_image_as_array(png_path)
            except Exception as exc:
                errors.append(f"[{folder}] Cannot open '{png_name}': {exc}")
                continue

            # Size check
            size_err = _check_image_size(arr, sz, png_name)
            if size_err:
                errors.append(f"[{folder}] {size_err}")

            # BaseColor: value-range check — max-min > 20 ensures visible
            # variation even if only two colours are present.
            if png_name == "BaseColor.png":
                color_range = int(arr.max()) - int(arr.min())
                if color_range <= BASECOLOR_MIN_RANGE:
                    errors.append(
                        f"[{folder}] 'BaseColor.png' value range is only "
                        f"{color_range} (max={int(arr.max())}, min={int(arr.min())}); "
                        f"need max-min > {BASECOLOR_MIN_RANGE} for visible variation"
                    )

            # Accumulate arrays and per-sample uniformity flag for later
            sample_maps.setdefault(png_name, []).append(arr)

    # ------------------------------------------------------------------
    # 4. Global uniformity gate
    #
    # A map PASSES uniformity if >80% of its samples are non-uniform.
    # If at least 2 maps pass  -> silently accept (one flat map is fine).
    # If fewer than 2 maps pass -> report every map that failed and why.
    # ------------------------------------------------------------------
    # per_map_uniform_samples: {map_name: [bool, ...]}  True = uniform (bad)
    per_map_uniform_samples = {}
    for map_name, arrays in sample_maps.items():
        per_map_uniform_samples[map_name] = [_is_uniform(a) for a in arrays]

    def _map_passes_uniformity(uniform_flags):
        """True if >80% of samples are NON-uniform."""
        if not uniform_flags:
            return False
        non_uniform_ratio = sum(not f for f in uniform_flags) / len(uniform_flags)
        return non_uniform_ratio > 0.80

    maps_passing_uniformity = [
        name for name, flags in per_map_uniform_samples.items()
        if _map_passes_uniformity(flags)
    ]

    if len(maps_passing_uniformity) < 2:
        # Not enough well-behaved maps — report each failing map
        for map_name, flags in per_map_uniform_samples.items():
            if not _map_passes_uniformity(flags):
                n_uniform = sum(flags)
                n_total = len(flags)
                pct = 100 * n_uniform / n_total if n_total else 0
                errors.append(
                    f"Uniformity: '{map_name}' is uniform in {n_uniform}/{n_total} "
                    f"samples ({pct:.0f}%) — exceeds the 20% tolerance "
                    f"and fewer than 2 maps overall pass the uniformity check"
                )

    # ------------------------------------------------------------------
    # 5. Global diversity gate  (only meaningful when numsamples > 1)
    #
    # A map PASSES diversity if >80% of its consecutive sample-pairs differ
    # by at least SAMPLE_DIFF_THRESHOLD fraction of pixels.
    # If at least 3 maps pass  -> silently accept (one static map is fine).
    # If fewer than 3 maps pass -> report every map that failed and why.
    # ------------------------------------------------------------------
    if numsamples > 1:
        # per_map_pair_results: {map_name: [(pair_label, frac_diff), ...]}
        per_map_pair_results = {}
        for map_name, arrays in sample_maps.items():
            if len(arrays) < 2:
                continue
            pairs = []
            for i in range(len(arrays) - 1):
                a, b = arrays[i], arrays[i + 1]
                if a.shape[:2] != b.shape[:2]:
                    continue  # size mismatch already flagged
                frac = _fraction_differing_pixels(a, b)
                pairs.append((f"sample_{i:03d} vs sample_{i+1:03d}", frac))
            per_map_pair_results[map_name] = pairs

        def _map_passes_diversity(pairs):
            """True if >80% of consecutive pairs have sufficient pixel difference."""
            if not pairs:
                return False
            passing = sum(1 for _, frac in pairs if frac >= SAMPLE_DIFF_THRESHOLD)
            return passing / len(pairs) > 0.80

        maps_passing_diversity = [
            name for name, pairs in per_map_pair_results.items()
            if _map_passes_diversity(pairs)
        ]

        if len(maps_passing_diversity) < 3:
            # Not enough well-behaved maps — report each failing map
            for map_name, pairs in per_map_pair_results.items():
                if not _map_passes_diversity(pairs):
                    failing = [
                        f"{label} ({frac*100:.2f}% differ)"
                        for label, frac in pairs
                        if frac < SAMPLE_DIFF_THRESHOLD
                    ]
                    errors.append(
                        f"Diversity: '{map_name}' has {len(failing)}/{len(pairs)} "
                        f"pairs that are too similar — exceeds the 20% tolerance "
                        f"and fewer than 3 maps overall pass the diversity check. "
                        f"Failing pairs: {'; '.join(failing)}"
                    )

    # ------------------------------------------------------------------
    # 6. Result
    # ------------------------------------------------------------------
    if errors:
        error_message = (
            "PBR verification FAILED with the following issues:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
        return True, error_message

    return False, "OK"


# ---------------------------------------------------------------------------
# Quick CLI usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python verify_pbr_output.py <outdir> [numsamples] [sz]")
        sys.exit(1)

    outdir_arg = sys.argv[1]
    numsamples_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    sz_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 512

    error_flag, message = verify_pbr_output(outdir_arg, numsamples_arg, sz_arg)
    print(f"Error: {error_flag}")
    print(message)
    sys.exit(1 if error_flag else 0)
