
# read content from file
def read_file_content(file_path: str) -> str:
    """
    Read the content of a file and return it as a string.
    
    Args:
        file_path: The path to the file to read.
    Returns:
        The content of the file as a string.
    """
    with open(file_path, 'r') as file:
        return file.read()