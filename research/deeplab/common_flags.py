default_configs = [{'type': 'integer', 'name': 'min_resize_value', 'value': None,
                    'description': 'Desired size of the smaller image side.'},
                   {'type': 'integer', 'name': 'max_resize_value', 'value': None,
                    'description': 'Maximum allowed size of the larger image side.'},
                   {'type': 'integer', 'name': 'resize_factor', 'value': None,
                    'description': 'Resized dimensions are multiple of factor plus one.'},
                   {'type': 'boolean', 'name': 'keep_aspect_ratio', 'value': True,
                    'description': 'Keep aspect ratio after resizing or not.'},
                   {'type': 'integer', 'name': 'logits_kernel_size', 'value': 1,
                    'description': 'The kernel size for the convolutional kernel that generates logits.'},
                   {'type': 'string', 'name': 'model_variant', 'value': 'mobilenet_v2',
                    'description': 'DeepLab model variant.'},
                   {'type': 'multi_float', 'name': 'image_pyramid', 'value': None,
                    'description': 'Input scales for multi-scale feature extraction.'},
                   {'type': 'boolean', 'name': 'add_image_level_feature', 'value': True,
                    'description': 'Add image level feature.'},
                   {'type': 'list', 'name': 'image_pooling_crop_size', 'value': None,
                    'description': 'Image pooling crop size [height, width] used in the ASPP module. '
                                   'When value is None, the model performs image pooling with "crop_size". Thisflag is useful when one likes to use different image pooling sizes.'},
                   {'type': 'list', 'name': 'image_pooling_stride', 'value': '1,1',
                    'description': 'Image pooling stride [height, width] used in the ASPP image pooling. '},
                   {'type': 'boolean', 'name': 'aspp_with_batch_norm', 'value': True,
                    'description': 'Use batch norm parameters for ASPP or not.'},
                   {'type': 'boolean', 'name': 'aspp_with_separable_conv', 'value': True,
                    'description': 'Use separable convolution for ASPP or not.'},
                   {'type': 'multi_integer', 'name': 'multi_grid', 'value': None,
                    'description': 'Employ a hierarchy of atrous rates for ResNet.'},
                   {'type': 'float', 'name': 'depth_multiplier', 'value': 1.0,
                    'description': 'Multiplier for the depth (number of channels) for all '
                                   'convolution ops used in MobileNet.'},
                   {'type': 'integer', 'name': 'divisible_by', 'value': None,
                    'description': 'An integer that ensures the layer # channels are divisible by this value. '
                                   'Used in MobileNet.'},
                   {'type': 'list', 'name': 'decoder_output_stride', 'value': None,
                    'description': 'Comma-separated list of strings with the number specifying output '
                                   'stride of low-level features at each network level.Current semantic '
                                   'segmentation implementation assumes at most one output stride (i.e., either '
                                   'None or a list with only one element.'},
                   {'type': 'boolean', 'name': 'decoder_use_separable_conv', 'value': True,
                    'description': 'Employ separable convolution for decoder or not.'},
                   {'type': 'enum', 'name': 'merge_method', 'value': ('max', ['max', 'avg']),
                    'description': 'Scheme to merge multi scale features.'},
                   {'type': 'boolean', 'name': 'prediction_with_upsampled_logits', 'value': True,
                    'description': 'When performing prediction, there are two options: (1) bilinear upsampling the '
                                   'logits followed by softmax, or (2) softmax followed by bilinear upsampling.'},
                   {'type': 'string', 'name': 'dense_prediction_cell_json', 'value': '',
                    'description': 'A JSON file that specifies the dense prediction cell.'},
                   {'type': 'integer', 'name': 'nas_stem_output_num_conv_filters', 'value': 20,
                    'description': 'Number of filters of the stem output tensor in NAS models.'},
                   {'type': 'bool', 'name': 'nas_use_classification_head', 'value': False,
                    'description': 'Use image classification head for NAS model variants.'},
                   {'type': 'bool', 'name': 'nas_remove_os32_stride', 'value': False,
                    'description': 'Remove the stride in the output stride 32 branch.'},
                   {'type': 'bool', 'name': 'use_bounded_activation', 'value': False,
                    'description': 'Whether or not to use bounded activations. Bounded activations better lend '
                                   'themselves to quantized inference.'},
                   {'type': 'boolean', 'name': 'aspp_with_concat_projection', 'value': True,
                    'description': 'ASPP with concat projection.'},
                   {'type': 'boolean', 'name': 'aspp_with_squeeze_and_excitation', 'value': False,
                    'description': 'ASPP with squeeze and excitation.'},
                   {'type': 'integer', 'name': 'aspp_convs_filters', 'value': 256,
                    'description': 'ASPP convolution filters.'},
                   {'type': 'boolean', 'name': 'decoder_use_sum_merge', 'value': False,
                    'description': 'Decoder uses simply sum merge.'},
                   {'type': 'integer', 'name': 'decoder_filters', 'value': 256, 'description': 'Decoder filters.'},
                   {'type': 'boolean', 'name': 'decoder_output_is_logits', 'value': False,
                    'description': 'Use decoder output as logits or not.'},
                   {'type': 'boolean', 'name': 'image_se_uses_qsigmoid', 'value': False,
                    'description': 'Use q-sigmoid.'},
                   {'type': 'multi_float', 'name': 'label_weights', 'value': None,
                    'description': 'A list of label weights, each element '
                                   'represents the weight for the label of its '
                                   'index, for example, label_weights = [0.1, 0.5] '
                                   'means the weight for label 0 is 0.1 and the '
                                   'weight for label 1 is 0.5. If set as None, all '
                                   'the labels have the same weight 1.0.'},
                   {'type': 'float', 'name': 'batch_norm_decay', 'value': 0.9997, 'description': 'Batchnorm decay.'}
                   ]
