
from cloudpickle.cloudpickle import instance
import string
import re
import os
import keras
import numpy as np
from keras import layers
from keras.utils import text_dataset_from_directory
from collections import Counter
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab

"""
PyTorch port of text_classification_from_scratch.py from keras-io
(@see: https://github.com/keras-team/keras-io/blob/master/examples/nlp/text_classification_from_scratch.py)

Has no TensorFlow dependence:
- Data loaded in "grain" format (needs grain to be installed)
- Uses torchtext for vocab & tokenizer

Imdb data source: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
"""

"""
## Setup
"""
os.nice(10) # Be nice!
os.environ["KERAS_BACKEND"] = "torch"
showSampleData = False

SPECIAL_TOKEN_UNK = "<unk>"
SPECIAL_TOKEN_PAD= "<pad>"
SPECIAL_TOKENS =[SPECIAL_TOKEN_UNK,SPECIAL_TOKEN_PAD]
SPECIAL_TOKEN_UNK_INDEX =0

max_features_incl_sp_tokens = 5000
batch_size = 32
embedding_dim = 128
sequence_length = 500
epochs = 3

# Actual max_features without special tokens = max tokens read from input files
max_features = max_features_incl_sp_tokens-len(SPECIAL_TOKENS)
imdb_files_dir="aclImdb"

#------ Start: Utils ----
def vectorize_text(input_data):
    i=0
    vectorzd_text_batch=[]    
    for text_batch in input_data:
        for text in text_batch:
            if(type(text)==str):
                vectorzd_text = [vocab[token] for token in tokenizer(custom_standardization(text))] 
                vectorzd_text.extend([SPECIAL_TOKEN_UNK_INDEX]*(vocabLength - len(vectorzd_text)))
                vectorzd_text_batch.append(vectorzd_text)
                i+=1
    vectorzd_text_batch=np.array(vectorzd_text_batch)
    vectorzd_text_batch= np.expand_dims(vectorzd_text_batch, -1)
    return [vectorzd_text_batch,input_data[1]]

def printSampleData(raw_data):
    for text_batch, label_batch in raw_data:
        for i in range(5):
            print(f"T{i} "+text_batch[i])
            print(f"B{i} "+str(label_batch[i]))
        break
    
def custom_standardization(inputText):
    return inputText.lower().replace("<br />", " ").replace(
        f"[{re.escape(string.punctuation)}]", ""
    )
#------ End: Utils ----

# Load Imdb data
raw_train_ds = text_dataset_from_directory(imdb_files_dir+"/train", batch_size=batch_size, 
                validation_split=0.2, seed=1337, subset="training", format="grain")
raw_val_ds = text_dataset_from_directory(imdb_files_dir+"/train", batch_size=batch_size,
                validation_split=0.2, seed=1337, subset="validation", format="grain")
raw_test_ds = text_dataset_from_directory(imdb_files_dir+"/test", batch_size=batch_size, format="grain")

if showSampleData: printSampleData(raw_train_ds)

print("Fetch Tokens")
tokens_train = []
tokenizer = get_tokenizer('basic_english')
for text_batch, label_batch in raw_train_ds:
    for text in text_batch:
        tokens_train.append(tokenizer(custom_standardization(text)))

print("Build Vocab")
counter = Counter()
for token in tokens_train: counter.update(token)
vocab = Vocab(counter, max_size=max_features-2)
vocabLength=len(vocab)
print("Vocab length: "+str(vocabLength))

assert vocab[tokenizer(SPECIAL_TOKEN_UNK)[0]]==SPECIAL_TOKEN_UNK_INDEX # Validate UNK token used for Padding

# Data mappers from raw data
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

print("Build model")

# A integer input for vocab indices.
inputs = keras.Input(shape=(None,), dtype="int64")

# Next, we add a layer to map those vocab indices into a space of dimensionality
# 'embedding_dim'.
x = layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)

# Conv1D + global max pooling
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)

# We add a vanilla hidden layer:
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# We project onto a single unit output layer, and squash it with a sigmoid:
predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

model = keras.Model(inputs, predictions)

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("Train model")
model.fit(val_ds, validation_data=val_ds, epochs=epochs)

print("Evaluate Trained Model")
model.evaluate(test_ds)
