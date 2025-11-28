import os
import math
import pytest
from keras.utils import text_dataset_from_directory
from TextClassificationTorchUtilities import *
from keras.src.utils.io_utils import print_msg
import TextClassificationTorchUtilities as tcu

os.nice(10)
BR="<br />"
INPUT = "This was Great"
EXCLAIM="!"
SPACE=" "
NEW_TOKEN = "NEW_TOKEN"

IMDB_TEN_FILES_DIR="tests/aclImdbTen" # Test folder with 10 files each
IMDB_ONE_FILES_DIR="tests/aclImdbOne" # Test folder with 1 file each
QUANTIZATION_MODE="int4"

def testPrintSample(): 
    raw_train_ds = text_dataset_from_directory(IMDB_TEN_FILES_DIR+"/train", batch_size=batch_size, 
                validation_split=0.2, seed=1337, subset="training", format="grain")
    assert raw_train_ds is not None
    assert raw_train_ds._batch_size == batch_size
    printSampleData(raw_train_ds)
    
def testCustomStandardization():

    assert SPACE+INPUT.lower()+SPACE == custom_standardization(BR+INPUT+EXCLAIM+BR)
    assert INPUT.lower()== custom_standardization(INPUT)

def testGetTokens():
    raw_train_ds = text_dataset_from_directory(IMDB_ONE_FILES_DIR+"/train", batch_size=batch_size, 
                validation_split=None, seed=1337, subset=None, format="grain")
    tokens = getTokens(raw_train_ds)
    assert tokens is not None
    assert len(tokens)==2
    assert len(tokens[0])==3
    assert len(tokens[1])==6
    
def testBuildVocab():

    raw_train_ds = text_dataset_from_directory(IMDB_ONE_FILES_DIR+"/train", batch_size=batch_size, 
                validation_split=None, seed=1337, subset=None, format="grain")
    tokens = getTokens(raw_train_ds)
    assert len(tokens)==2

    expedctedVocabSize = 7 + len(SPECIAL_TOKENS)
    vocab = buildVocab(tokens)
    print(vocab)
    
    assert len(vocab) == expedctedVocabSize
    
    # Test rebuild vocab
    assert len(tokens[0])==3
    tokens[0].remove("great")
    assert len(tokens[0])==2
    
    # Build & update cached vocab (1 token less)
    vocab = buildVocab(tokens)
    assert len(vocab) == expedctedVocabSize - 1
    
    # Cached vocab returned (1 token less)
    vocab = buildVocabFromRawData(raw_train_ds)
    assert len(vocab) == expedctedVocabSize -1
    
    # Rebuild vocab, tokens reloaded, missing token restored
    vocab = buildVocabFromRawData(raw_train_ds,rebuild=True)
    assert len(vocab) == expedctedVocabSize
    
    # Save vocab
    saveVocabPath=SAVE_TO_DIR+"testOneVocabTextClassfn.pkl"
    vocab1 = buildVocabFromRawData(raw_train_ds,rebuild=True,saveVocabPath=saveVocabPath)
    
    vocab2 = loadVocabFromFile(saveVocabPath)
    print(vocab2)
    assert vocab1 == vocab2
        
    vocab1 = loadVocabFromFile(SAVE_TO_DIR+"TextClassificationVocab1.pkl")
    vocab2 = loadVocabFromFile(SAVE_TO_DIR+"TextClassificationVocab2.pkl")
    vocab3 = loadVocabFromFile(SAVE_TO_DIR+"TextClassificationVocab3.pkl")
    assert vocab1!= vocab and vocab1 == vocab2 and vocab1 == vocab3

def testVectorizeText():
    raw_train_ds = text_dataset_from_directory(IMDB_ONE_FILES_DIR+"/train", batch_size=batch_size, 
                validation_split=None, seed=1337, subset=None, format="grain")
    
    buildVocabFromRawData(raw_train_ds)
    
    train_ds = raw_train_ds.map(vectorize_text)
    vectorizeTextBatch =next(iter(train_ds))
    
    assert vectorizeTextBatch is not None
    assert len(vectorizeTextBatch)==2
    assert vectorizeTextBatch[0].shape ==(2,9,1)
    assert len(vectorizeTextBatch[1]) ==2
    # Verify Padding
    assert str(vectorizeTextBatch[0][0].T)== "[[2 3 5 0 0 0 0 0 0]]" 
    assert str(vectorizeTextBatch[0][1].T)== "[[2 3 6 4 8 7 0 0 0]]"

modelSummary=""
def printFunc(summary):
    global modelSummary
    modelSummary+=summary
    
def testBuildModel():
    raw_train_ds = text_dataset_from_directory(IMDB_ONE_FILES_DIR+"/train", batch_size=batch_size, 
                validation_split=None, seed=1337, subset=None, format="grain")
    train_ds = raw_train_ds.map(vectorize_text)
    
    model=buildCompileModel(train_ds, train_ds, train_ds, max_features, embedding_dim, epochs)
    
    keras.utils.plot_model(model, USER_HOME+"/Tools/models/saved/TextClassificationTorch/TextClassificationTorchModel_Full.png", 
        show_dtype=True,show_shapes=True,show_layer_activations=True, show_trainable=True,show_layer_names=True)
    
    model.summary(show_trainable=True)
    
    assert model.compiled
    assert len(model.layers)==9
    assert len(model.operations) ==9
    assert model.loss=="binary_crossentropy"
    assert len(model.metrics[1]._user_metrics)==1 and model.metrics[1]._user_metrics[0] =="accuracy"

def testLoadAndPlotModel():
    model=keras.models.load_model(SAVE_TO_DIR+'TextClassificationStudentModel2.5Kc.keras')
    keras.utils.plot_model(model, USER_HOME+"/Tools/models/saved/TextClassificationTorch/TextClassificationStudentModel2.5Kc.png", 
        show_dtype=True,show_shapes=True,show_layer_activations=True, show_trainable=True,show_layer_names=True)
    model.summary(show_trainable=True)

def testModelQuantization():
    quantizeAndValidate()
    testEvaluateModel()

def testEvaluateModel():
    print("Load Vocab")
    loadVocabFromFile(SAVE_TO_DIR+"TextClassificationVocab.pkl")
    
    model=keras.models.load_model(SAVE_TO_DIR+'TextClassificationTorchModel.keras')

    vectorzd_text_batch,one_batch_labels = vectorize_text([["This was a great movie!",
                "Horrible is the word"],[]])
    evalRes = model(vectorzd_text_batch)
    assert math.isclose(1.0, evalRes[0][0].item(), abs_tol=0.3)
    assert math.isclose(0.0, evalRes[1][0].item(), abs_tol=0.3)
  
def quantizeAndValidate():
    raw_train_ds = text_dataset_from_directory(IMDB_ONE_FILES_DIR+"/train", batch_size=batch_size, 
                validation_split=None, seed=1337, subset=None, format="grain")
    train_ds = raw_train_ds.map(vectorize_text)
    
    model=buildCompileModel(train_ds, train_ds, train_ds, max_features, embedding_dim, epochs)
    assert model.layers[1].quantization_mode is None

    model.quantize(QUANTIZATION_MODE)
    assert model.layers[1].quantization_mode == QUANTIZATION_MODE
    keras.utils.plot_model(model, USER_HOME+"/Tools/models/saved/TextClassificationTorch/TextClassificationTorchModel_q4.png", 
        show_dtype=True,show_shapes=True,show_layer_activations=True, show_trainable=True,show_layer_names=True)
    
    model.summary(show_trainable=True)

    assert model.compiled
    assert len(model.layers)==9
    assert len(model.operations) ==9
    assert model.loss=="binary_crossentropy"
    assert len(model.metrics[1]._user_metrics)==1 and model.metrics[1]._user_metrics[0] =="accuracy"
    
def testQuantizeAndSaveModel():
    model=keras.models.load_model(SAVE_TO_DIR+'TextClassificationTorchModel.keras')
    model.quantize(QUANTIZATION_MODE)
    model.save(SAVE_TO_DIR+'TextClassificationTorchModel_q4.keras')
    assert True 
  