import os
from PIL import Image
import base64
import io
def check_and_create_directory(dir_path):
    """Check if a directory exists, and if not, create it.
    
    Args:
    - dir_path: The path of the directory to check and create if necessary.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
      
