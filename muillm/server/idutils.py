import secrets
import string

def generate_id(prefix: str = "", length: int = 6) -> str:
    """
    Generate a random alphanumeric ID with optional prefix and length control.
    
    Args:
        prefix (str): Optional string to prepend to the ID.
        length (int): Length of the random part.
    
    Returns:
        str: Generated unique ID.
    """
    alphabet = string.ascii_letters + string.digits  # A-Z, a-z, 0-9
    random_part_length = length
    random_part = ''.join(secrets.choice(alphabet) for _ in range(random_part_length))
    return prefix + random_part