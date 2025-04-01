from pathlib import Path

import dill
import numpy as np
import yaml


def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.

    :param file_path: Path to the input YAML file.
    :return: Dictionary with the YAML file content.
    """
    path = Path(file_path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes the given `content` (e.g., dict) to a YAML file.

    :param file_path: Path to the output file.
    :param content: Content to write to the file.
    :param replace: Whether to replace the file if it already exists.
    """
    path = Path(file_path)
    if replace and path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(content), encoding="utf-8")


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Saves a NumPy array to a binary file.

    :param file_path: Path to the output file.
    :param array: NumPy array to save.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_obj:
        np.save(file_obj, array)


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Loads a NumPy array from a binary file.

    :param file_path: Path to the input file.
    :return: The loaded NumPy array.
    """
    path = Path(file_path)
    with path.open("rb") as file_obj:
        return np.load(file_obj)


def save_object(file_path: str, obj: object) -> None:
    """
    Serializes a Python object to a binary file using dill.

    :param file_path: Path to the output file.
    :param obj: The Python object to save.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_obj:
        dill.dump(obj, file_obj)


def load_object(file_path: str) -> object:
    """
    Deserializes a Python object from a binary file using dill.

    :param file_path: Path to the input file.
    :return: The deserialized Python object.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"The {file_path} doesn't exist.")
    with path.open("rb") as file_obj:
        return dill.load(file_obj)
