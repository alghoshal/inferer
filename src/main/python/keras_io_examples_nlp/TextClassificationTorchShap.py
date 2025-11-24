from TextClassificationTorchUtilities import *
import shap
from shap import maskers

"""
Perform Inference using a saved Keras PyTorch TextClassification model and its vocabulary (vocab)

Imdb data source: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
"""

"""
## Setup
"""
    
imdb_files_dir="tests/aclImdbTen" # Ten files 
batch_size=2
# Load Imdb data
raw_test_ds = text_dataset_from_directory(imdb_files_dir+"/test", batch_size=batch_size, format="grain")
test_ds = raw_test_ds.map(vectorize_text)

one_batch_text=[]
# Fetch one batch data from test_ds
for text_batch, label_batch in raw_test_ds:  
    one_batch_text.append(text_batch)

one_batch_text=np.array(one_batch_text)
   
print("Load Vocab")
vocab=loadVocabFromFile(SAVE_TO_DIR+"TextClassificationVocab.pkl")

print("Load & Evaluate: ")
keraModel=keras.models.load_model(SAVE_TO_DIR+'TextClassificationTorchModel.keras')

def predict(vectorizedData):
    print("Predict called")
    outputs = keraModel(vectorizedData).detach().numpy()
    #print(outputs)
    return outputs

def vectorizer(mask, *args):
    print("Vectorizer called")
    vectorzd_text_batch,one_batch_labels = vectorize_text([args[0],[]])
    return vectorzd_text_batch
    
explainer = shap.Explainer(predict,masker=vectorizer)
shap_values = explainer(one_batch_text, batch_size=batch_size, max_evals=10)
print(shap_values)

#shap.plots.text(shap_values[0])