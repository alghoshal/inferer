from TextClassificationTorchUtilities import *
import shap
from shap import maskers
import random


"""
View Shap values of a trained Keras PyTorch TextClassification 
(Model: ./text_classification_torch.py)

Imdb data source: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
"""

"""
## Setup
"""
    
imdb_files_dir="tests/aclImdbTen" # Ten files 
batch_size=5
SHAP_PLOT_OUTPUT_FILE=SAVE_TO_DIR+'TextClassificationShapPlot.html'
readDataFiles=False
maxReviewsToLoad=3

if readDataFiles:
    # Load Imdb data
    raw_test_ds = text_dataset_from_directory(imdb_files_dir+"/test", batch_size=batch_size, format="grain")
    # Load just the 1st review from every batch
    one_batch_text=raw_test_ds.map(lambda x: x[0][0]) 
else:
    one_batch_text=["This is a great one to watch.","What a long drawn boring affair to the end credits."]
    one_batch_text=np.array(one_batch_text)

print("Load Vocab")
vocab=loadVocabFromFile(SAVE_TO_DIR+"TextClassificationVocab.pkl")

print("Load Model")
kerasModel=keras.models.load_model(SAVE_TO_DIR+'TextClassificationTorchModel.keras')

'''
Input data has words masked (at random) using the specified mask(SPECIAL_TOKEN_UNK). 
E.g.: "<unk> is a great one to watch.", "This is a <unk> one to watch.",...

Snap uses the output to evaluate the impact of the specific masked word (token).
'''
def predict(inputData):
    print("Predict called: "+inputData)
    vectorzd_text_batch,one_batch_labels = vectorize_text([inputData,[]])
    outputs = kerasModel(vectorzd_text_batch).detach().numpy().flatten()
    print(outputs)
    return outputs

def custom_tokenizer(s, return_offsets_mapping=True):
    """Custom tokenizers conform to a subset of the transformers API."""
    pos = 0
    offset_ranges = []
    input_ids = []
    for m in re.finditer(r"\W", s):
        start, end = m.span(0)
        offset_ranges.append((pos, start))
        input_ids.append(s[pos:start])
        pos = end
    if pos != len(s):
        offset_ranges.append((pos, len(s)))
        input_ids.append(s[pos:])
    out = {}
    out["input_ids"] = input_ids
    if return_offsets_mapping:
        out["offset_mapping"] = offset_ranges
    return out

masker = maskers.Text(custom_tokenizer, mask_token=SPECIAL_TOKEN_UNK)
explainer = shap.Explainer(predict,masker=masker)
shap_values = explainer(one_batch_text, batch_size=batch_size, max_evals=5)
print(shap_values)

shapPlot=shap.plots.text(shap_values,display=False)

# Save shapPlot
with open(SHAP_PLOT_OUTPUT_FILE, "w") as f:
    f.write(shapPlot)