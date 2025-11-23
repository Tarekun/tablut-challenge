import argparse
import glob
import os
import re


def rescale(source_bounds: tuple, target_bounds: tuple, value: float) -> float:
    source_max, source_min = source_bounds
    source_range = source_max - source_min
    target_max, target_min = target_bounds
    target_range = target_max - target_min

    normalized = (value - source_min) / source_range
    return target_min + normalized * target_range


def flag_manager():
    parse = argparse.ArgumentParser()
    parse.add_argument("-res", action="store_true", help="Activate Residual Connection")
    args = parse.parse_args()
    return args


def get_latest_checkpoint(
    folder: str, prefix: str = "tablut_model_checkpoint_iter_", ext: str = ".pth"
):
    pattern = os.path.join(folder, f"{prefix}*{ext}")
    files = glob.glob(pattern)
    if not files:
        return None, 0

    def extract_iter_number(path):
        match = re.search(r"iter_(\d+)\.pth$", path)
        return int(match.group(1)) if match else -1

    latest_file = max(files, key=extract_iter_number)

    return latest_file, extract_iter_number(latest_file)
