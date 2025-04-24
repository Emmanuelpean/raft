"""Module containing general utility functions"""

from operator import itemgetter

import numpy as np


def sort_lists(
    alist: list | tuple | np.ndarray,
    *indices: int,
) -> np.ndarray:
    """Sort a list/tuple of np.ndarrays with respect to the ndarrays corresponding to the indices given
    :param alist: list/tuple of ndarrays of the same size
    :param indices: indices"""

    if not indices:
        indices = (0,)
    # noinspection PyArgumentList
    data_sorted = sorted(zip(*alist), key=itemgetter(*indices))
    return np.transpose(data_sorted)


def normalise_list(data: any) -> any:
    """Normalises various list inputs into a consistent format:
    - If input is a list of objects: return as-is.
    - If input is a list of lists of objects, return a list of objects
    - If input is a list of dicts of objects: return dict of lists of objects.
    - If input is a list of dicts of lists of objects: return dict of lists of objects."""

    first = data[0]

    # List of dictionaries
    if isinstance(first, dict):
        result = {}
        keys = {key for d in data for key in d}

        for key in keys:
            result[key] = normalise_list([d[key] for d in data if key in d])
        return result

    # List of lists
    elif isinstance(first, list):
        result = []
        for d in data:
            result += d
        return result

    else:
        return data


def make_unique(alist: list[str]) -> list[str]:
    """Make the element of a list of strings unique
    :param alist: list of strings"""

    new_list = []
    for string in alist:
        if string not in new_list:
            new_list.append(string)
        else:
            i = 1
            while True:
                new_string = string + f" ({i})"
                if new_string in new_list:
                    i += 1
                else:
                    new_list.append(new_string)
                    break

    return new_list


def merge_dicts(*dictionaries: dict) -> dict:
    """Recursively merge multiple dictionaries, keeping the first occurrence of each key.
    If a key appears in more than one dictionary, the value from the first dictionary containing the key will be used.
    """

    merged_dictionary = {}
    for dictionary in dictionaries:
        for key in dictionary.keys():

            # If they key already exist but the two are both dictionaries, merge them
            if key in merged_dictionary and isinstance(merged_dictionary[key], dict) and isinstance(dictionary[key], dict):
                merged_dictionary[key] = merge_dicts(merged_dictionary[key], dictionary[key])

            # If the key is not already present, add it
            if key not in merged_dictionary:
                merged_dictionary[key] = dictionary[key]

    return merged_dictionary
