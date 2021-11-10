import yaml


def config_deep_sort(config):
    with open(config) as f:
        return yaml.load(f, Loader=yaml.FullLoader)
