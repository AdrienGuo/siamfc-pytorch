import yaml


class Config():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, yaml_path):
        # Load config.yaml
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
            self.__dict__.update(cfg)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            yaml.safe_dump(self.__dict__, f, sort_keys=False)

    def update(self, json_path):
        """Loads parameters from json file."""
        with open(json_path) as f:
            cfg = yaml.safe_load(f)
            self.__dict__.update(cfg)

    def update_with_dict(self, dictio):
        """Updates the parameters with the keys and values of a dictionary."""
        self.__dict__.update(dictio)
