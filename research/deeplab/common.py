# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides flags that are common to scripts.

Common flags from train/eval/vis/export_model.py are collected in this script.
"""
import collections
import copy
import json
import tensorflow as tf
from deeplab import common_flags
from deeplab.configloader import ConfigLoader

flags = tf.app.flags
FLAGS = flags.FLAGS


def handle_type(entry):
    etype = entry['type']
    name = entry['name']
    value = entry['value']
    desc = entry['description']
    if etype == "float":
        flags.DEFINE_float(name, value, desc)
    if etype == "boolean":
        flags.DEFINE_boolean(name, value, desc)
    if etype == "bool":
        flags.DEFINE_bool(name, value, desc)
    if etype == "multi_integer":
        flags.DEFINE_multi_integer(name, value, desc)
    if etype == "string":
        flags.DEFINE_string(name, value, desc)
    if etype == "enum":
        flags.DEFINE_enum(name, value[0], value[1], desc)
    if etype == "list":
        flags.DEFINE_list(name, value, desc)
    if etype == "integer":
        flags.DEFINE_integer(name, value, desc)


for entry in common_flags.default_configs:
    handle_type(entry)

# Constants

# Perform semantic segmentation predictions.
OUTPUT_TYPE = 'semantic'

# Semantic segmentation item names.
LABELS_CLASS = 'labels_class'
IMAGE = 'image'
HEIGHT = 'height'
WIDTH = 'width'
IMAGE_NAME = 'image_name'
LABEL = 'label'
ORIGINAL_IMAGE = 'original_image'

# Test set name.
TEST_SET = 'test'

cl = ConfigLoader(FLAGS=FLAGS)


class ModelOptions(
    collections.namedtuple('ModelOptions', [
        'outputs_to_num_classes',
        'crop_size',
        'atrous_rates',
        'output_stride',
        'preprocessed_images_dtype',
        'merge_method',
        'add_image_level_feature',
        'image_pooling_crop_size',
        'image_pooling_stride',
        'aspp_with_batch_norm',
        'aspp_with_separable_conv',
        'multi_grid',
        'decoder_output_stride',
        'decoder_use_separable_conv',
        'logits_kernel_size',
        'model_variant',
        'depth_multiplier',
        'divisible_by',
        'prediction_with_upsampled_logits',
        'dense_prediction_cell_config',
        'nas_architecture_options',
        'use_bounded_activation',
        'aspp_with_concat_projection',
        'aspp_with_squeeze_and_excitation',
        'aspp_convs_filters',
        'decoder_use_sum_merge',
        'decoder_filters',
        'decoder_output_is_logits',
        'image_se_uses_qsigmoid',
        'label_weights',
        'sync_batch_norm_method',
        'batch_norm_decay',
    ])):
    """Immutable class to hold model options."""

    __slots__ = ()

    def __new__(cls,
                outputs_to_num_classes=None,
                crop_size=None,
                atrous_rates=None,
                output_stride=8,
                preprocessed_images_dtype=tf.float32):
        """Constructor to set default values.

        Args:
          outputs_to_num_classes: A dictionary from output type to the number of
            classes. For example, for the task of semantic segmentation with 21
            semantic classes, we would have outputs_to_num_classes['semantic'] = 21.
          crop_size: A tuple [crop_height, crop_width].
          atrous_rates: A list of atrous convolution rates for ASPP.
          output_stride: The ratio of input to output spatial resolution.
          preprocessed_images_dtype: The type after the preprocessing function.

        Returns:
          A new ModelOptions instance.
        """
        dense_prediction_cell_config = None
        if cl.dense_prediction_cell_json:
            with tf.gfile.Open(cl.dense_prediction_cell_json, 'r') as f:
                dense_prediction_cell_config = json.load(f)
        decoder_output_stride = None
        if cl.decoder_output_stride:
            decoder_output_stride = [
                int(x) for x in cl.decoder_output_stride]
            if sorted(decoder_output_stride, reverse=True) != decoder_output_stride:
                raise ValueError('Decoder output stride need to be sorted in the '
                                 'descending order.')
        image_pooling_crop_size = None
        if cl.image_pooling_crop_size:
            image_pooling_crop_size = [int(x) for x in cl.image_pooling_crop_size]
        image_pooling_stride = [1, 1]
        if cl.image_pooling_stride:
            image_pooling_stride = [int(x) for x in cl.image_pooling_stride]
        label_weights = cl.label_weights
        if label_weights is None:
            label_weights = 1.0
        nas_architecture_options = {
            'nas_stem_output_num_conv_filters': (
                cl.nas_stem_output_num_conv_filters),
            'nas_use_classification_head': cl.nas_use_classification_head,
            'nas_remove_os32_stride': cl.nas_remove_os32_stride,
        }
        return super(ModelOptions, cls).__new__(
            cls, outputs_to_num_classes, crop_size, atrous_rates, output_stride,
            preprocessed_images_dtype,
            cl.merge_method,
            cl.add_image_level_feature,
            image_pooling_crop_size,
            image_pooling_stride,
            cl.aspp_with_batch_norm,
            cl.aspp_with_separable_conv,
            cl.multi_grid,
            decoder_output_stride,
            cl.decoder_use_separable_conv,
            cl.logits_kernel_size,
            cl.model_variant,
            cl.depth_multiplier,
            cl.divisible_by,
            cl.prediction_with_upsampled_logits,
            dense_prediction_cell_config,
            nas_architecture_options,
            cl.use_bounded_activation,
            cl.aspp_with_concat_projection,
            cl.aspp_with_squeeze_and_excitation,
            cl.aspp_convs_filters,
            cl.decoder_use_sum_merge,
            cl.decoder_filters,
            cl.decoder_output_is_logits,
            cl.image_se_uses_qsigmoid,
            label_weights,
            'None',
            cl.batch_norm_decay)

    def __deepcopy__(self, memo):
        return ModelOptions(copy.deepcopy(self.outputs_to_num_classes),
                            self.crop_size,
                            self.atrous_rates,
                            self.output_stride,
                            self.preprocessed_images_dtype)
