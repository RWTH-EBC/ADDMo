import os
import shutil

def root_dir():
    """
    Finds the root directory of the git repository.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def create_or_clean_directory(path: str) -> str:
    """
    Creates a directory or optionally deletes its contents if it exists.
    """
    if not os.path.exists(path):
        # Path does not exist, create it
        os.makedirs(path)
    elif not os.listdir(path):
        # Path exists, but is empty
        pass
    else:
        # Path exists, ask for confirmation to delete current contents
        print(f"The directory {path} already exists and contains the following files/folders:")
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            print(f" - {file_path}")
        response = input("To overwrite the content type <y>, for deleting the current contents type <d>: ")
        if response.lower() == 'd':
            # Delete the contents of the directory
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        elif response.lower() == 'y':
            pass
        else:
            print("Operation cancelled.")
            return None

    return path


def create_path_or_ask_to_override(filename, directory, override: bool = True) -> str:
    """
    Creates a file path and optionally overwrites the existing file.
    """
    path = create_dir_and_get_path(filename, directory)
    _overwrite_file(path, override)
    return path


def create_dir_and_get_path(filename: str, directory: str) -> str:
    """
    Returns the full path for a given filename and directory.
    """
    if directory is not None:  # check if directory is none
        if not os.path.exists(directory):  # check if path exists
            os.makedirs(directory)  # create new directory
        return os.path.join(directory, filename)  # calculate full path
    else:
        return filename


def _overwrite_file(path: str, overwrite: bool):
    """
    Checks if a file exists and if it should be overwritten.
    """
    if os.path.exists(path) and not overwrite:
        if not _get_bool(
                f'The file "{path}" already exists. Do you want to override it?\n'
        ):
            return 0


def _get_bool(message: str, true: list = None, false: list = None) -> bool or str:
    """
    Gets a boolean value from the user.
    """
    if false is None:
        false = ["no", "nein", "false", "1", "n"]
    if true is None:
        true = ["yes", "ja", "true", "wahr", "0", "y"]

    val = input(message).lower()
    if val in true:
        return True
    elif val in false:
        return False
    else:
        print("Please try again.")
        print("True:", true)
        print("False:", false)
        _get_bool(message, true, false)

    return val


