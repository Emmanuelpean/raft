"""Module containing functions for comparing objects"""

import numpy as np


def are_identical(
    obj1: any,
    obj2: any,
    rtol: float | None = None,
) -> bool:
    """Check if two objects have identical values. This does not check if the types are the same
    :param obj1: list or dictionary to compare.
    :param obj2: list or dictionary to compare.
    :param rtol: Relative tolerance for floating-point comparisons using np.allclose."""

    if isinstance(obj1, dict) and isinstance(obj2, dict):
        if obj1.keys() != obj2.keys():
            return False
        else:
            return all(are_identical(obj1[k], obj2[k], rtol) for k in obj1)

    elif isinstance(obj1, (list, tuple, np.ndarray)) and isinstance(obj2, (list, tuple, np.ndarray)):
        if len(obj1) != len(obj2):
            return False
        else:
            return all(are_identical(i1, i2, rtol) for i1, i2 in zip(obj1, obj2))

    else:
        if isinstance(obj1, float) and isinstance(obj2, float):
            if rtol is not None:
                return np.allclose(obj1, obj2, rtol=rtol, equal_nan=True)
            else:
                return np.array_equal(obj1, obj2, equal_nan=True)

        else:
            return obj1 == obj2


def are_close(*args, rtol=1e-3) -> bool:
    """Check if two objects are similar"""

    return are_identical(*args, rtol=rtol)
