"""Module containing general utility functions"""

import tomllib
from operator import itemgetter
from pathlib import Path
import datetime as dt

import numpy as np
import requests


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
            if (
                key in merged_dictionary
                and isinstance(merged_dictionary[key], dict)
                and isinstance(dictionary[key], dict)
            ):
                merged_dictionary[key] = merge_dicts(merged_dictionary[key], dictionary[key])

            # If the key is not already present, add it
            if key not in merged_dictionary:
                merged_dictionary[key] = dictionary[key]

    return merged_dictionary


def get_pyproject_info(*keys: str) -> any:
    """Get information from the pyproject file"""

    pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    d = data[keys[0]]
    for key in keys[1:]:
        d = d[key]
    return d


def get_last_commit_date_from_github(
    repo_url: str,
    branch: str = "main",
) -> str:
    """Get the date of the latest commit
    :param repo_url: repository url
    :param branch: specific branch"""

    # Extract the owner and repo name from the URL
    repo_parts = repo_url.rstrip("/").split("/")[-2:]
    owner, repo = repo_parts[0], repo_parts[1]

    # GitHub API endpoint to fetch the latest commit
    api_url = f"https://api.github.com/repos/{owner}/{repo}/commits?sha={branch}"

    try:
        # Send the request to GitHub API
        response = requests.get(api_url)
        response.raise_for_status()  # Raise error for bad status codes

        # Extract the last commit date from the JSON response
        commit_data = response.json()[0]
        commit_date = commit_data["commit"]["committer"]["date"]
        date = dt.datetime.strptime(commit_date, "%Y-%m-%dT%H:%M:%SZ")
        return date.strftime("%d %B %Y")
    except requests.exceptions.RequestException as e:
        return f"Error fetching data: {e}"
