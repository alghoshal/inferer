import pytest
from keras.utils import text_dataset_from_directory
from TextClassificationTorchUtilities import *
import TextClassificationTorchUtilities as tcu

BR="<br />"
INPUT = "This was Great"
EXCLAIM="!"
SPACE=" "
NEW_TOKEN = "NEW_TOKEN"

IMDB_TEN_FILES_DIR="tests/aclImdbTen" # Test folder with 10 files each
IMDB_ONE_FILES_DIR="tests/aclImdbOne" # Test folder with 1 file each


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
    
    # Rebuild vocab, tokens reloaded, missing token restrored
    vocab = buildVocabFromRawData(raw_train_ds,rebuild=True)
    assert len(vocab) == expedctedVocabSize

def testVectorizeText():
    raw_train_ds = text_dataset_from_directory(IMDB_ONE_FILES_DIR+"/train", batch_size=batch_size, 
                validation_split=None, seed=1337, subset=None, format="grain")
    
    buildVocabFromRawData(raw_train_ds)
    
    # Fetch one_batch_data
    batchOneText = []
    batchOneLabel=None
    for text_batch, label_batch in raw_train_ds:
        for text in text_batch:
            batchOneText.append(text)
        batchOneLabel=label_batch
        break
    
    vectorizeTextBatch = vectorize_text((batchOneText,batchOneLabel))
    
    assert vectorizeTextBatch is not None
    assert len(vectorizeTextBatch)==2
    assert vectorizeTextBatch[0].shape ==(2,9,1)
    assert len(vectorizeTextBatch[1]) ==2
    # Verify Padding
    assert str(vectorizeTextBatch[0][0].T)== "[[2 3 5 0 0 0 0 0 0]]" 
    assert str(vectorizeTextBatch[0][1].T)== "[[2 3 6 4 8 7 0 0 0]]"


def testBuildModel():
    raw_train_ds = text_dataset_from_directory(IMDB_ONE_FILES_DIR+"/train", batch_size=batch_size, 
                validation_split=None, seed=1337, subset=None, format="grain")
    train_ds = raw_train_ds.map(vectorize_text)
    
    model=buildCompileModel(train_ds, train_ds, train_ds, max_features, embedding_dim, epochs)
    assert model.compiled
    assert len(model.layers)==9
    assert len(model.operations) ==9
    assert model.loss=="binary_crossentropy"
    assert len(model.metrics[1]._user_metrics)==1 and model.metrics[1]._user_metrics[0] =="accuracy"
    