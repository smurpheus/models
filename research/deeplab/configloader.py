import os, json


class ConfigLoader(object):
    FLAGS = None
    DEFAULT_FLAGS = None
    def __init__(self, config_path=None):
        configs = self.load_config(config_path)
        for key, value in configs.items():
            setattr(self, key, value)

    def load_config(self, config_path=None):
        if config_path and os.path.isfile(config_path):
            with open(config_path) as jsonfile_handle:
                configs = json.load(jsonfile_handle)
                self.DEFAULT_FLAGS.update(configs)
                return self.DEFAULT_FLAGS
        else:
            configs = dict(atrous_rates=self.FLAGS.atrous_rates,
                           output_stride=self.FLAGS.output_stride,
                           image_pyramid=self.FLAGS.image_pyramid,
                           weight_deacay=self.FLAGS.weight_decay,
                           fine_tune_batch_norm=self.FLAGS.fine_tune_batch_norm,
                           drop_path_keep_prob=self.FLAGS.drop_path_keep_prob,
                           training_number_of_steps=self.FLAGS.training_number_of_steps,
                           upsample_logits=self.FLAGS.upsample_logits,
                           hard_example_mining_step=self.FLAGS.hard_example_mining_step,
                           top_k_percent_pixels=self.FLAGS.top_k_percent_pixels,
                           )
            return configs
