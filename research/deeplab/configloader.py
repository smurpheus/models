import os, json


class ConfigLoader(object):
    FLAGS = None
    DEFAULT_FLAGS = None
    def __init__(self, config_path=None, FLAGS=FLAGS):
        self.FLAGS = FLAGS
        configs = self.load_config(config_path)
        for key, value in configs.items():
            if type(value) == tuple:
              setattr(self, key, value[0])
            else:
              setattr(self, key, value)

    def load_config(self, config_path=None):
        if config_path and os.path.isfile(config_path):
            with open(config_path) as jsonfile_handle:
                configs = json.load(jsonfile_handle)
                self.DEFAULT_FLAGS.update(configs)
                return self.DEFAULT_FLAGS
        else:
            configs = {key: self.FLAGS[key].value for key in self.FLAGS}
            return configs
