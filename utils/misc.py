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


def new_parameter(name, IV, hopt=True, config=True, models=True, hopt_type=None, hopt_range=None):
    upper = name.upper()
    lower = name.lower()
    if hopt:
        if hopt_type is not None and hopt_range is not None:
            htype = hopt_type.lower()
            print("HOPT LINE:")
            if htype == 'categorical':
                print(f"'{lower}': trial.suggest_{htype}('{lower}', {hopt_range}),")
            else:
                print(f"'{lower}': trial.suggest_{htype}('{lower}', {hopt_range[0]}, {hopt_range[1]}),")
            print("##########")
    if config:
        print("CONFIG LINE:")
        print(f"{upper} = {IV}")
        print("##########")
    if models:
        print("MODELS LINE:")
        print(f"self.{lower} = kwargs.get('{lower}', config.{upper})")
        print("##########")
    return print("COPY AND PASTE ABOVE")

