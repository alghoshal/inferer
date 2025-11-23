import pytest
from keras.utils import text_dataset_from_directory
from TextClassificationTorchUtilities import *

BR="<br />"
INPUT = "This was Great"
EXCLAIM="!"
SPACE=" "

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

    vocab = buildVocab(tokens)
    print(vocab)
    assert len(vocab) == 7 + len(SPECIAL_TOKENS)

