import os
import pickle


def read_pkl(filename: str, directory: str = 'storedData'):
    """
    Reads data from a pickle file.
    :param filename: The name of the file.
    :param directory: The directory the file is located.
    :return: Pickle object.
    """

    path = _get_path(filename, directory)

    if os.path.exists(path):                                # check for the existence of the path
        pkl_file = open(path, 'rb')                         # open path
        pkl_data = pickle.load(pkl_file)                    # read data
        pkl_file.close()                                    # close path

        if pkl_data is not None:
            return pkl_data                                     # return data
        else:
            raise FileNotFoundError(f'No data at {path} found.')
    else:
        raise FileNotFoundError(f'The path {path} does not exist.')


def write_pkl(data, filename: str, directory: str = 'storedData', override: bool = False):
    """
    Writes data to a pickle file.
    :param data: The object that is supposed to be saved.
    :param filename: The name of the file.
    :param directory: The directory the file will be saved to.
    :param override: Boolean.
    """

    path = _get_path(filename, directory)

    # make sure the directory does exist
    if directory is not None:
        if not os.path.exists(directory):
            os.mkdir(directory)

    # make sure the file does not already exist
    if os.path.exists(path) and not override:
        if not _get_bool(f'The file "{path}" already exists. Do you want to override it?\n'):
            return 0

    # open the path and writhe the file
    pkl_file = open(path, 'wb')
    pickle.dump(data, pkl_file)

    # close file
    pkl_file.close()


def _get_bool(message: str,
              true: list = ['yes', 'ja', 'true', 'wahr', '0', 'y'],
              false: list = ['no', 'nein', 'false', '1', 'n']):

    val = input(message).lower()
    if val in true:
        return True
    elif val in false:
        return False
    else:
        print('Please try again.')
        print('True:', true)
        print('False:', false)
        _get_bool(message, true, false)

    return val
