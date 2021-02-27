# Lint as: python2, python3
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

"""Converts PASCAL VOC 2012 data to TFRecord file format with Example protos.

PASCAL VOC 2012 dataset is expected to have the following directory structure:

  + pascal_voc_seg
    - build_data.py
    - build_voc2012_data.py (current working directory).
    + VOCdevkit
      + VOC2012
        + JPEGImages
        + SegmentationClass
        + ImageSets
          + Segmentation
    + tfrecord

Image folder:
  ./VOCdevkit/VOC2012/JPEGImages

Semantic segmentation annotations:
  ./VOCdevkit/VOC2012/SegmentationClass

list folder:
  ./VOCdevkit/VOC2012/ImageSets/Segmentation

This script converts data into sharded data files and save at tfrecord folder.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os.path
import sys

from deeplab.configloader import ConfigLoader
from deeplab.datasets import build_data
from six.moves import range
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
flags = tf.app.flags
default_configs = [{'type': 'string', 'name': 'image_folder', 'value': './VOCdevkit/VOC2012/JPEGImages',
                    'description': 'Folder containing images.'},
                   {'type': 'string', 'name': 'semantic_segmentation_folder',
                    'value': './VOCdevkit/VOC2012/SegmentationClassRaw',
                    'description': 'Folder containing semantic segmentation annotations.'},
                   {'type': 'string', 'name': 'list_folder', 'value': './VOCdevkit/VOC2012/ImageSets/Segmentation',
                    'description': 'Folder containing lists for training and validation'},
                   {'type': 'string', 'name': 'output_dir', 'value': './tfrecord',
                    'description': 'Path to save converted SSTable of TensorFlow examples.'}]
configs = {entry['name']: entry['value'] for entry in default_configs}


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


for entry in default_configs:
    handle_type(entry)

_NUM_SHARDS = 4


class Datasetbuilder(ConfigLoader):
    FLAGS = FLAGS
    DEFAULT_FLAGS = configs

    def _convert_dataset(self, dataset_split):
        """Converts the specified dataset split to TFRecord format.

        Args:
          dataset_split: The dataset split (e.g., train, test).

        Raises:
          RuntimeError: If loaded image and label have different shape.
        """
        dataset = os.path.basename(dataset_split)[:-4]
        sys.stdout.write('Processing ' + dataset)
        filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
        num_images = len(filenames)
        num_per_shard = int(math.ceil(num_images / _NUM_SHARDS))

        image_reader = build_data.ImageReader('jpeg', channels=3)
        label_reader = build_data.ImageReader('png', channels=1)

        for shard_id in range(_NUM_SHARDS):
            output_filename = os.path.join(
                self.output_dir,
                '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                start_idx = shard_id * num_per_shard
                end_idx = min((shard_id + 1) * num_per_shard, num_images)
                for i in range(start_idx, end_idx):
                    sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                        i + 1, len(filenames), shard_id))
                    sys.stdout.flush()
                    # Read the image.
                    image_filename = os.path.join(
                        self.image_folder, filenames[i] + '.' + self.image_format)
                    image_data = tf.gfile.GFile(image_filename, 'rb').read()
                    height, width = image_reader.read_image_dims(image_data)
                    # Read the semantic segmentation annotation.
                    seg_filename = os.path.join(
                        self.semantic_segmentation_folder,
                        filenames[i] + '.' + self.label_format)
                    seg_data = tf.gfile.GFile(seg_filename, 'rb').read()
                    seg_height, seg_width = label_reader.read_image_dims(seg_data)
                    if height != seg_height or width != seg_width:
                        raise RuntimeError('Shape mismatched between image and label.')
                    # Convert to tf example.
                    example = build_data.image_seg_to_tfexample(
                        image_data, filenames[i], height, width, seg_data)
                    tfrecord_writer.write(example.SerializeToString())
            sys.stdout.write('\n')
            sys.stdout.flush()

    def build_dataset(self):
        dataset_splits = tf.gfile.Glob(os.path.join(self.list_folder, '*.txt'))
        for dataset_split in dataset_splits:
            self._convert_dataset(dataset_split)


def main(unused_argv):
    db = Datasetbuilder()
    db.build_dataset()


if __name__ == '__main__':
    tf.app.run()
