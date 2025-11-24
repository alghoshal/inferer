from TextClassificationTorchUtilities import *

"""
Perform Inference using a saved Keras PyTorch TextClassification model and its vocabulary (vocab)

Imdb data source: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
"""

"""
## Setup
"""

#imdb_files_dir="aclImdbTen"
# Load Imdb data
raw_test_ds = text_dataset_from_directory(imdb_files_dir+"/test", batch_size=batch_size, format="grain")

print("Load Vocab")
loadVocabFromFile(SAVE_TO_DIR+"TextClassificationVocab.pkl")

# Data mappers from raw data
test_ds = raw_test_ds.map(vectorize_text)

print("Load & Evaluate Full Model")
model=keras.models.load_model(SAVE_TO_DIR+'TextClassificationTorchModel.keras')
model.evaluate(test_ds)

print("Load & Evaluate Quantized Model")
model=keras.models.load_model(SAVE_TO_DIR+'TextClassificationTorchModel_q4.keras')
model.evaluate(test_ds)