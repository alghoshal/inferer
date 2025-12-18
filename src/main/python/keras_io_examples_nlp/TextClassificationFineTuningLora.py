from TextClassificationTorchUtilities import *
import os

"""
Fine tunes a Keras PyTorch TextClassification model using LoRA adapters.

The original model text_classification_torch.py is a Sentiments classifier
trained on the Imdb reviews dataset which labels data as either 1 (Positive) or 0 (Negative).

This model fine tunes the earlier model using LoRA adapters on the same dataset 
to turn it into an Exaggeration detector (classfier). It labels the data as either
1 (Has-Exaggerations) or 0 (No Exaggerations).

Exaggerations are any reviews containing the exaggerations defined in: 
TextClassificationTorchUtilities.exaggerations

Imdb data source: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
"""

"""
## Setup
"""
os.nice(10)  # Be nice!
imdb_files_dir = "tests/aclImdb"
# Load Imdb data
raw_train_ds = text_dataset_from_directory(imdb_files_dir+"/train", batch_size=batch_size,
                                           validation_split=0.2, seed=1337, subset="training", format="grain")
raw_val_ds = text_dataset_from_directory(imdb_files_dir+"/train", batch_size=batch_size,
                                         validation_split=0.2, seed=1337, subset="validation", format="grain")
raw_test_ds = text_dataset_from_directory(
    imdb_files_dir+"/test", batch_size=batch_size, format="grain")

# Data mappers from raw data
train_ds = raw_train_ds.map(buildExaggerationDataset).map(vectorize_text)
val_ds = raw_val_ds.map(buildExaggerationDataset).map(vectorize_text)
test_ds = raw_test_ds.map(buildExaggerationDataset).map(vectorize_text)


print("Load Vocab")
loadVocabFromFile(SAVE_TO_DIR+"TextClassificationVocab.pkl")

print("Load Model")
modelPath = SAVE_TO_DIR+'TextClassificationTorchModel.keras'
model = keras.models.load_model(modelPath)

print("Enable Lora adapters")
enableLora(model)

model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False),
              optimizer=keras.optimizers.Adam(), metrics=[keras.metrics.BinaryAccuracy()])

print("Train model")
model.fit(train_ds, validation_data=val_ds, epochs=epochs)

print("Evaluate Trained Model")
model.evaluate(test_ds)

print("Save Model")
model.save(SAVE_TO_DIR+'TextClassificationModelFineTunedLoraExaggerations.keras')
