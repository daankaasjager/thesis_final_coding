from typing import List


def read_list(path: str) -> List[str]:
    """
    Reads a list of strings from a file, stripping whitespace and ignoring empty lines. Delimits with comma

    Args:
        path (str): The file path to read from.

    Returns:
        List[str]: A list of strings read from the file.
    """
    with open(path, "r") as f:
        content = f.read()
    return [line.strip() for line in content.split(",") if line.strip()]
