import glob
import os


def clear_local_ckpt_files():
    """
    Clears all .ckpt files in cwd
    :return: None
    """
    ckpt_files = glob.glob("*.ckpt")

    for file in ckpt_files:
        try:
            os.remove(file)
        except OSError:
            pass  # For when mutliple machines running in parallel try to remove same file
        print(f"Deleted checkpoint file: {file}")
    return None
