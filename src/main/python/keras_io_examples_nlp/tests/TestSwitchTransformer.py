import os
import math
import pytest
from TextClassificationTorchUtilities import *
import TextClassificationTorchUtilities as tcu
from SwitchTransformerUtil import *

os.nice(10)
BR="<br />"
INPUT = "This was Great"
EXCLAIM="!"
SPACE=" "
NEW_TOKEN = "NEW_TOKEN"

IMDB_TEN_FILES_DIR="aclImdbTen" # Test folder with 10 files each
IMDB_ONE_FILES_DIR="aclImdbOne" # Test folder with 1 file each
QUANTIZATION_MODE="int4"
embedding_dim=32

modelSummary=""

raw_train_ds=[(["This was a great movie!","Horrible is the word"],np.array([1,0]))]
batch_size=len(raw_train_ds[0][0])
num_tokens_per_example=-1
num_tokens_per_batch=-1

def callTokenPosnEmbedding():
    global batch_size,num_tokens_per_example
    vocab = buildVocabFromRawData(raw_train_ds,rebuild=True)
    vocabLength=len(vocab)
    vectorzd_text_batch,one_batch_labels = vectorize_text(raw_train_ds[0])
   
    tokenAndPositionEmbedding = TokenAndPositionEmbedding(vocabLength, vocabLength, embed_dim)
    tokenPosnEmbeded=tokenAndPositionEmbedding.call(vectorzd_text_batch)
    batch_size = ops.shape(tokenPosnEmbeded)[0]
    num_tokens_per_example = ops.shape(tokenPosnEmbeded)[1]
    return tokenPosnEmbeded

def callRouter():
    tokenPosnEmbeded = callTokenPosnEmbedding()
    inputs = ops.reshape(tokenPosnEmbeded, [batch_size * num_tokens_per_example, embed_dim])
    router = Router(num_experts, expert_capacity=tcu.vocabLength)
    return router.call(inputs)

def testTokenAndPositionEmbedding():
    assert callTokenPosnEmbedding().shape == (len(raw_train_ds[0][0]),tcu.vocabLength,1,embed_dim)

def testRouter():
    dispatcher, combiner = callRouter()
    assert dispatcher.shape == combiner.shape and dispatcher.shape == (
        batch_size * num_tokens_per_example, num_experts, batch_size * num_tokens_per_example // num_experts)

def testSwitch():
    global num_tokens_per_batch, num_tokens_per_example
    tokenPosnEmbeded = callTokenPosnEmbedding()
    num_tokens_per_batch=batch_size*num_tokens_per_example

    # Switch
    switch = Switch(num_experts, embed_dim, ff_dim, num_tokens_per_batch)
    outputSwitch = switch.call(tokenPosnEmbeded)
    assert outputSwitch.shape ==(batch_size, num_tokens_per_example, embed_dim)
    
    # Simple Switch
    simpleSwitchRoute = SimpleSwitchRoute(num_experts, embed_dim, ff_dim, num_tokens_per_batch)
    simpleSwitchRoute.gate = switch.router.route # Reuse gate
    outputSimpleSwitch = simpleSwitchRoute.call(tokenPosnEmbeded)
    assert outputSimpleSwitch.shape ==(batch_size, num_tokens_per_example, embed_dim)
    
 #   assert np.all(ops.nonzero(outputSwitch).eq(ops.nonzero(outputSimpleSwitch)).tolist())

testSwitch()