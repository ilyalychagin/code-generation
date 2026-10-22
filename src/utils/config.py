from runpy import run_path


def load_config(path):
    namespace = run_path(str(path))
    return namespace["Config"]()
