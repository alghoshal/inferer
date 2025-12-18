"""
PyTorch port of text_classification_with_switch_transformer.py from keras-io
(@see: https://github.com/keras-team/keras-io/blob/master/examples/nlp/text_classification_with_switch_transformer.py)

Changed
- super.init call in Router and Switch
- position_in_expert recomputed with updated expert_mask in Router->call()
- Modularized

See [Switch Transformer](https://arxiv.org/abs/2101.03961) which uses Mixture of Expert (MoE) routing layer
in place of FFN in Transformers.
"""

"""
## Introduction

This example demonstrates the implementation of the
[Switch Transformer](https://arxiv.org/abs/2101.03961) model for text
classification.

The Switch Transformer replaces the feedforward network (FFN) layer in the standard
Transformer with a Mixture of Expert (MoE) routing layer, where each expert operates
independently on the tokens in the sequence. This allows increasing the model size without
increasing the computation needed to process each example.

Note that, for training the Switch Transformer efficiently, data and model parallelism
need to be applied, so that expert modules can run simultaneously, each on its own accelerator.
While the implementation described in the paper uses the
[TensorFlow Mesh](https://github.com/tensorflow/mesh) framework for distributed training,
this example presents a simple, non-distributed implementation of the Switch Transformer
model for demonstration purposes.
"""

"""
## Setup
"""


"""
## Download and prepare dataset
"""

from SwitchTransformerUtil import *
from keras import layers
from keras import ops
import os
import keras  #
os.nice(10)  # Be nice!
os.environ["KERAS_BACKEND"] = "torch"

(x_train, y_train), (x_val, y_val) = fetchN(
    keras.datasets.imdb.load_data(num_words=vocab_size), N=samplesCount)

print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.utils.pad_sequences(x_train, maxlen=num_tokens_per_example)
x_val = keras.utils.pad_sequences(x_val, maxlen=num_tokens_per_example)

print(f"Number of tokens per batch: {num_tokens_per_batch}")


classifier = create_classifier(useSimpleSwitch=True)
run_experiment(classifier, x_train, y_train, x_val, y_val)
