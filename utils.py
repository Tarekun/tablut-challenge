def rescale(source_bounds: tuple, target_bounds: tuple, value: float) -> float:
    source_max, source_min = source_bounds
    source_range = source_max - source_min
    target_max, target_min = target_bounds
    target_range = target_max - target_min

    normalized = (value - source_min) / source_range
    return target_min + normalized * target_range
