from TextClassificationTorchUtilities import *

"""
Fine tunes a Keras PyTorch TextClassification model using LoRA adapters

Imdb data source: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
"""

"""
## Setup
"""

imdb_files_dir="aclImdbTen"
# Load Imdb data
# raw_test_ds = text_dataset_from_directory(imdb_files_dir+"/test", batch_size=batch_size, format="grain")
# test_ds = raw_test_ds.map(vectorize_text)

print("Load Vocab")
loadVocabFromFile(SAVE_TO_DIR+"TextClassificationVocab.pkl")

print("Load Model")
modelPath=SAVE_TO_DIR+'TextClassificationTorchModel.keras'
model=keras.models.load_model(modelPath)
        
model.summary(show_trainable=True)
enableLora(model,layerNames=["conv1d_1","dense"])
model.summary(show_trainable=True)