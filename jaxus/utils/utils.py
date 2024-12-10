import numpy as np


def fix_extent(extent):
    """Ensures that the extent is in the form (x0, x1, z0, z1)"""
    e0, e1, e2, e3 = extent
    return np.array([min(e0, e1), max(e0, e1), min(e2, e3), max(e2, e3)])


def interpret_range(range_str, dim_size):
    """Interprets a range string"""
    if range_str is None or "all" in range_str or range_str == "-1":
        return list(range(dim_size))

    if "-" in range_str:
        start, end = range_str.split("-")
        return list(range(int(start), int(end) + 1))

    # Remove all commas and brackets
    range_str = range_str.replace("[", " ").replace("]", " ").replace(",", " ")

    return list(map(int, range_str.split()))
