'''
Using Evidently, identify Drift between review text and sentiment labels from two very distinct datasets:
- Movies review (Imdb): https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
- Code review: https://github.com/evidentlyai/evidently/blob/main/examples/datasets/code_review.csv

@author algo
'''

import pandas as pd
import os
from evidently.metrics import DriftedColumnsCount
from evidently.metrics import ValueDrift
from evidently import Dataset, DataDefinition
from evidently import Report
from evidently.generators import ColumnMetricGenerator
from evidently.metrics import UniqueValueCount
from evidently.descriptors import TextLength,WordCount
from evidently.presets import TextEvals
import csv
from TextClassificationTorchUtilities import *
from keras.utils import text_dataset_from_directory

USER_HOME=os.path.expanduser("~")
SAVE_TO_DIR_EVIDENTLY=USER_HOME+"/Tools/models/saved/Evidently/"
IMDB_TEN_FILES_DIR="datasets/aclImdbTen" 

os.nice(10) # Be nice!
os.environ["KERAS_BACKEND"] = "torch"

def getCurrData():
    reference=pd.read_csv("datasets/code_review.csv")
    refSentiments = reference.filter(items=["Expert label"]).map(lambda x: 0 if x=="bad" else 1)
    reference = reference.filter(items=["Generated review"]).join(refSentiments)
    reference.rename(columns={"Generated review":"Review","Expert label": "Label"}, inplace=True)
    return reference
   
def getRefData():
    imdbReviewsRaw = text_dataset_from_directory(IMDB_TEN_FILES_DIR+"/train", batch_size=50, 
                validation_split=None, seed=1337, subset=None, format="grain")
    imdbReviews = next(iter(imdbReviewsRaw))
    return pd.DataFrame({"Review":imdbReviews[0],"Label":imdbReviews[1]})

def getRefExpertSentiments(reference):
    loadVocabFromFile(SAVE_TO_DIR+"TextClassificationVocab.pkl")
    model=keras.models.load_model(SAVE_TO_DIR+'TextClassificationTorchModel.keras')
    refExpertComments = reference.filter(items=["Expert comment"]).to_numpy().flatten()
    vectorzd_text_batch,labelsVectorized = vectorize_text((refExpertComments,[]))
    evalRes = model(vectorzd_text_batch)
    return evalRes

def runSaveReport(driftReport, current, reference, filePath):
    snapshot = driftReport.run(current, reference)
    snapshot.save_html(filePath)
    
def buildDatasetWithDescriptors(df):
    dataset = Dataset.from_pandas(
        df
    )
    dataset.add_descriptors(descriptors=[
        TextLength("Review", alias="LengthReview"),
        WordCount("Review", alias="WordCountReview")
    ])
    return dataset

def generateDriftReport(reference, current):
    reference = buildDatasetWithDescriptors(reference)
    current = buildDatasetWithDescriptors(current)
    runSaveReport(Report([
        TextEvals(),
        ValueDrift(column="LengthReview", method="psi", threshold=0.05),     
        ValueDrift(column="WordCountReview", method="kl_div"),
        DriftedColumnsCount(cat_stattest="psi", num_stattest="wasserstein", per_column_method={"Label":"psi"}, drift_share=0.5),
        ],
        include_tests=False), current, reference, filePath=SAVE_TO_DIR_EVIDENTLY+"/driftSnapshotTextReviews.html")

    runSaveReport(Report([
        ValueDrift(column="Review", method="perc_text_content_drift"),
        ValueDrift(column="Label", method="chisquare"),
        ],
        include_tests=False), current, reference, filePath=SAVE_TO_DIR_EVIDENTLY+"/driftSnapshotTextReviews1.html")
    
    runSaveReport(Report([
        ColumnMetricGenerator(ValueDrift, columns=['LengthReview', 'WordCountReview']),         
        ColumnMetricGenerator(UniqueValueCount, column_types="cat"),]), 
        current, reference, filePath=SAVE_TO_DIR_EVIDENTLY+"/genDriftSnapshotTextReviews.html")
 

reference = getRefData()
current = getCurrData()
generateDriftReport(reference,current)