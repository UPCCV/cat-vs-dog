#!/usr/bin/env bash

cd ~/CNN/tensorflow
bazel build tensorflow/examples/label_image/...

cd ~/models
bazel build tensorflow/python/tools:freeze_graph
bazel build tensorflow/tools/graph_transforms:summarize_graph
bazel build tensorflow/tools/graph_transforms:transform_graph
bazel build tensorflow/tools/quantization:quantize_graph
bazel build tensorflow/contrib/util:convert_graphdef_memmapped_format