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
"""Training script for the DeepLab model.

See model.py for more details and usage.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import six
from deeplab import train_flags
import tensorflow as tf
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib import tfprof as contrib_tfprof
from deeplab import common
from deeplab import model
from deeplab.datasets import data_generator
from deeplab.utils import train_utils
from slim.deployment import model_deploy
from deeplab.configloader import ConfigLoader

slim = tf.contrib.slim
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


for entry in train_flags.default_configs:
    handle_type(entry)

print(FLAGS)

class Trainer(ConfigLoader):
    FLAGS = FLAGS
    DEFAULT_FLAGS = train_flags.configs
    def _build_deeplab(self, iterator, outputs_to_num_classes, ignore_label):
        """Builds a clone of DeepLab.

        Args:
          iterator: An iterator of type tf.data.Iterator for images and labels.
          outputs_to_num_classes: A map from output type to the number of classes. For
            example, for the task of semantic segmentation with 21 semantic classes,
            we would have outputs_to_num_classes['semantic'] = 21.
          ignore_label: Ignore label.
        """
        samples = iterator.get_next()

        # Add name to input and label nodes so we can add to summary.
        samples[common.IMAGE] = tf.identity(samples[common.IMAGE], name=common.IMAGE)
        samples[common.LABEL] = tf.identity(samples[common.LABEL], name=common.LABEL)

        model_options = common.ModelOptions(
            outputs_to_num_classes=outputs_to_num_classes,
            crop_size=[int(sz) for sz in self.train_crop_size],
            atrous_rates=self.atrous_rates,
            output_stride=self.output_stride)

        outputs_to_scales_to_logits = model.multi_scale_logits(
            samples[common.IMAGE],
            model_options=model_options,
            image_pyramid=common.cl.image_pyramid,
            weight_decay=self.weight_decay,
            is_training=True,
            fine_tune_batch_norm=self.fine_tune_batch_norm,
            nas_training_hyper_parameters={
                'drop_path_keep_prob': self.drop_path_keep_prob,
                'total_training_steps': self.training_number_of_steps,
            })

        # Add name to graph node so we can add to summary.
        output_type_dict = outputs_to_scales_to_logits[common.OUTPUT_TYPE]
        output_type_dict[model.MERGED_LOGITS_SCOPE] = tf.identity(
            output_type_dict[model.MERGED_LOGITS_SCOPE], name=common.OUTPUT_TYPE)

        for output, num_classes in six.iteritems(outputs_to_num_classes):
            train_utils.add_softmax_cross_entropy_loss_for_each_scale(
                outputs_to_scales_to_logits[output],
                samples[common.LABEL],
                num_classes,
                ignore_label,
                loss_weight=model_options.label_weights,
                upsample_logits=self.upsample_logits,
                hard_example_mining_step=self.hard_example_mining_step,
                top_k_percent_pixels=self.top_k_percent_pixels,
                scope=output)

    def do_training(self):
        # Set up deployment (i.e., multi-GPUs and/or multi-replicas).
        config = model_deploy.DeploymentConfig(
            num_clones=self.num_clones,
            clone_on_cpu=self.clone_on_cpu,
            replica_id=self.task,
            num_replicas=self.num_replicas,
            num_ps_tasks=self.num_ps_tasks)

        # Split the batch across GPUs.
        assert self.train_batch_size % config.num_clones == 0, (
            'Training batch size not divisble by number of clones (GPUs).')

        clone_batch_size = self.train_batch_size // config.num_clones

        tf.gfile.MakeDirs(self.train_logdir)
        tf.logging.info('Training on %s set', self.train_split)

        with tf.Graph().as_default() as graph:
            with tf.device(config.inputs_device()):
                dataset = data_generator.Dataset(
                    dataset_name=self.dataset,
                    split_name=self.train_split,
                    dataset_dir=self.dataset_dir,
                    batch_size=clone_batch_size,
                    crop_size=[int(sz) for sz in self.train_crop_size],
                    min_resize_value=self.min_resize_value,
                    max_resize_value=self.max_resize_value,
                    resize_factor=self.resize_factor,
                    min_scale_factor=self.min_scale_factor,
                    max_scale_factor=self.max_scale_factor,
                    scale_factor_step_size=self.scale_factor_step_size,
                    model_variant=self.model_variant,
                    num_readers=4,
                    is_training=True,
                    should_shuffle=True,
                    should_repeat=True)

            # Create the global step on the device storing the variables.
            with tf.device(config.variables_device()):
                global_step = tf.train.get_or_create_global_step()

                # Define the model and create clones.
                model_fn = self._build_deeplab
                model_args = (dataset.get_one_shot_iterator(), {
                    common.OUTPUT_TYPE: dataset.num_of_classes
                }, dataset.ignore_label)
                clones = model_deploy.create_clones(config, model_fn, args=model_args)

                # Gather update_ops from the first clone. These contain, for example,
                # the updates for the batch_norm variables created by model_fn.
                first_clone_scope = config.clone_scope(0)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

            # Gather initial summaries.
            summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

            # Add summaries for model variables.
            for model_var in tf.model_variables():
                summaries.add(tf.summary.histogram(model_var.op.name, model_var))

            # Add summaries for images, labels, semantic predictions
            if self.save_summaries_images:
                summary_image = graph.get_tensor_by_name(
                    ('%s/%s:0' % (first_clone_scope, common.IMAGE)).strip('/'))
                summaries.add(
                    tf.summary.image('samples/%s' % common.IMAGE, summary_image))

                first_clone_label = graph.get_tensor_by_name(
                    ('%s/%s:0' % (first_clone_scope, common.LABEL)).strip('/'))
                # Scale up summary image pixel values for better visualization.
                pixel_scaling = max(1, 255 // dataset.num_of_classes)
                summary_label = tf.cast(first_clone_label * pixel_scaling, tf.uint8)
                summaries.add(
                    tf.summary.image('samples/%s' % common.LABEL, summary_label))

                first_clone_output = graph.get_tensor_by_name(
                    ('%s/%s:0' % (first_clone_scope, common.OUTPUT_TYPE)).strip('/'))
                predictions = tf.expand_dims(tf.argmax(first_clone_output, 3), -1)

                summary_predictions = tf.cast(predictions * pixel_scaling, tf.uint8)
                summaries.add(
                    tf.summary.image(
                        'samples/%s' % common.OUTPUT_TYPE, summary_predictions))

            # Add summaries for losses.
            for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
                summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

            # Build the optimizer based on the device specification.
            with tf.device(config.optimizer_device()):
                learning_rate = train_utils.get_model_learning_rate(
                    self.learning_policy,
                    self.base_learning_rate,
                    self.learning_rate_decay_step,
                    self.learning_rate_decay_factor,
                    self.training_number_of_steps,
                    self.learning_power,
                    self.slow_start_step,
                    self.slow_start_learning_rate,
                    decay_steps=self.decay_steps,
                    end_learning_rate=self.end_learning_rate)

                summaries.add(tf.summary.scalar('learning_rate', learning_rate))

                if self.optimizer == 'momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate, self.momentum)
                elif self.optimizer == 'adam':
                    optimizer = tf.train.AdamOptimizer(
                        learning_rate=self.adam_learning_rate, epsilon=self.adam_epsilon)
                else:
                    raise ValueError('Unknown optimizer')

            if self.quantize_delay_step >= 0:
                if self.num_clones > 1:
                    raise ValueError('Quantization doesn\'t support multi-clone yet.')
                contrib_quantize.create_training_graph(
                    quant_delay=self.quantize_delay_step)

            startup_delay_steps = self.task * self.startup_delay_steps

            with tf.device(config.variables_device()):
                total_loss, grads_and_vars = model_deploy.optimize_clones(
                    clones, optimizer)
                total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')
                summaries.add(tf.summary.scalar('total_loss', total_loss))

                # Modify the gradients for biases and last layer variables.
                last_layers = model.get_extra_layer_scopes(
                    self.last_layers_contain_logits_only)
                grad_mult = train_utils.get_model_gradient_multipliers(
                    last_layers, self.last_layer_gradient_multiplier)
                if grad_mult:
                    grads_and_vars = slim.learning.multiply_gradients(
                        grads_and_vars, grad_mult)

                # Create gradient update op.
                grad_updates = optimizer.apply_gradients(
                    grads_and_vars, global_step=global_step)
                update_ops.append(grad_updates)
                update_op = tf.group(*update_ops)
                with tf.control_dependencies([update_op]):
                    train_tensor = tf.identity(total_loss, name='train_op')

            # Add the summaries from the first clone. These contain the summaries
            # created by model_fn and either optimize_clones() or _gather_clone_loss().
            summaries |= set(
                tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))

            # Merge all summaries together.
            summary_op = tf.summary.merge(list(summaries))

            # Soft placement allows placing on CPU ops without GPU implementation.
            session_config = tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False)

            # Start the training.
            profile_dir = self.profile_logdir
            if profile_dir is not None:
                tf.gfile.MakeDirs(profile_dir)

            with contrib_tfprof.ProfileContext(
                    enabled=profile_dir is not None, profile_dir=profile_dir):
                init_fn = None
                if self.tf_initial_checkpoint:
                    init_fn = train_utils.get_model_init_fn(
                        self.train_logdir,
                        self.tf_initial_checkpoint,
                        self.initialize_last_layer,
                        last_layers,
                        ignore_missing_vars=True)

                slim.learning.train(
                    train_tensor,
                    logdir=self.train_logdir,
                    log_every_n_steps=self.log_steps,
                    master=self.master,
                    number_of_steps=self.training_number_of_steps,
                    is_chief=(self.task == 0),
                    session_config=session_config,
                    startup_delay_steps=startup_delay_steps,
                    init_fn=init_fn,
                    summary_op=summary_op,
                    save_summaries_secs=self.save_summaries_secs,
                    save_interval_secs=self.save_interval_secs)


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    t = Trainer(FLAGS=FLAGS)
    t.do_training()


if __name__ == '__main__':
    flags.mark_flag_as_required('train_logdir')
    flags.mark_flag_as_required('dataset_dir')
    tf.app.run()
