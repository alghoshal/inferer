from TextClassificationTorchUtilities import *

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

#imdb_files_dir="aclImdbTen"
# Load Imdb data
raw_train_ds = text_dataset_from_directory(imdb_files_dir+"/train", batch_size=batch_size, 
                validation_split=0.2, seed=1337, subset="training", format="grain")
raw_val_ds = text_dataset_from_directory(imdb_files_dir+"/train", batch_size=batch_size,
                validation_split=0.2, seed=1337, subset="validation", format="grain")
raw_test_ds = text_dataset_from_directory(imdb_files_dir+"/test", batch_size=batch_size, format="grain")

if showSampleData: printSampleData(raw_train_ds)

print("Build Vocab")
buildVocabFromRawData(raw_train_ds,saveVocabPath=SAVE_TO_DIR+'TextClassificationVocab.pkl')

# Data mappers from raw data
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

print("Build model")
model=buildCompileModel(train_ds, val_ds, test_ds, max_features, embedding_dim, epochs)

print("Train model")
model.fit(train_ds, validation_data=val_ds, epochs=epochs)

print("Evaluate Trained Model")
model.evaluate(test_ds)

print("Save Model")
model.save(SAVE_TO_DIR+'TextClassificationTorchModel.keras')