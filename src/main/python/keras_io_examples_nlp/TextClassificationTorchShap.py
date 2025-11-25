from TextClassificationTorchUtilities import *
import shap
from shap import maskers
import random

"""
Perform Inference using a saved Keras PyTorch TextClassification model and its vocabulary (vocab)

Imdb data source: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
"""

"""
## Setup
"""
    
imdb_files_dir="tests/aclImdbTen" # Ten files 
batch_size=1
# Load Imdb data
#raw_test_ds = text_dataset_from_directory(imdb_files_dir+"/test", batch_size=batch_size, format="grain")

one_batch_text=[("This is great one to watch.",)]
# Fetch one batch data from test_ds
#for text_batch, label_batch in raw_test_ds:  
#    one_batch_text.append(text_batch)

one_batch_text=np.array(one_batch_text)
   
print("Load Vocab")
vocab=loadVocabFromFile(SAVE_TO_DIR+"TextClassificationVocab.pkl")

print("Load Model")
kerasModel=keras.models.load_model(SAVE_TO_DIR+'TextClassificationTorchModel.keras')

def predict(vectorizedData):
    print("Predict called ")
    vectorzd_text_batch,one_batch_labels = vectorize_text([vectorizedData,[]])
    outputs = kerasModel(vectorzd_text_batch).detach().numpy().flatten()
    print(outputs)
    return outputs

# TODO: Fix me!
def maskIt(mask,text):
    maskedText = ["This is great one to watch. ",
             "This is <unk> one to watch. ",
            "<unk> is great one to watch. ",
            "<unk> <unk> <unk> one to watch. ",
            "<unk> <unk> <unk> <unk> to watch. ",
            "<unk> <unk> <unk> <unk> <unk> watch. ",
            "<unk> <unk> <unk> <unk> <unk> <unk> "
            ]
    random.shuffle(maskedText)
    return maskedText[:3]
counter=0
def masker(mask, *args):
    global counter
    counter+=1
    print("Masker called")
    return maskIt(mask,args[0])
    
explainer = shap.Explainer(predict,masker=masker)
shap_values = explainer(one_batch_text[:1], batch_size=batch_size, max_evals=5)
print(shap_values[0])

shapPlot=shap.plots.text(shap_values[0],display=False)
print(shapPlot)