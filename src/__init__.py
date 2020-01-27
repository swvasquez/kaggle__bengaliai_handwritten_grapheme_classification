import pathlib


def project_paths():
    """Allows files in module know where to find important project paths."""
    project_root = pathlib.Path(__file__).parents[1]
    config_path = project_root / 'config.yaml'
    data_path = project_root / 'data'
    paths = {'config': config_path, 'data': data_path, 'root': project_root}

    return paths
