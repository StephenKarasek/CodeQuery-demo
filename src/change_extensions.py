import os
import sys


def change_extensions(root_dir, old_exts, new_ext):
    """
    Walk through the directory tree starting at root_dir and rename files with
    extensions in old_exts to have the new_ext extension.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(old_exts):
                old_path = os.path.join(dirpath, filename)
                # Split the filename into the base and its extension, then add the new extension
                base, _ = os.path.splitext(filename)
                new_filename = base + new_ext
                new_path = os.path.join(dirpath, new_filename)

                print(f"Renaming: {old_path} -> {new_path}")
                os.rename(old_path, new_path)


if __name__ == '__main__':
    # Check if a directory was provided as a command-line argument; otherwise use the current directory.
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = './TEST/CeetronAccess'

    # Define the extensions to search for and the new extension to apply.
    extensions_to_change = ('.h', '.cs', '.cmake')
    new_extension = '.txt'

    change_extensions(directory, extensions_to_change, new_extension)
