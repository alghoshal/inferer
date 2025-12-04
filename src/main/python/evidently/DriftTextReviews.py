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
from evidently import Report
from evidently.generators import ColumnMetricGenerator
from evidently.metrics import UniqueValueCount
import csv
from TextClassificationTorchUtilities import *
from keras.utils import text_dataset_from_directory

USER_HOME=os.path.expanduser("~")
SAVE_TO_DIR_EVIDENTLY=USER_HOME+"/Tools/models/saved/Evidently/"
IMDB_TEN_FILES_DIR="datasets/aclImdbTen" 

os.nice(10) # Be nice!
os.environ["KERAS_BACKEND"] = "torch"

def getRefData():
    reference=pd.read_csv("datasets/code_review.csv")
    refSentiments = reference.filter(items=["Expert label"]).map(lambda x: 0 if x=="bad" else 1)
    return reference.filter(items=["Generated review"]).join(refSentiments)
   
def getCurrData():
    imdbReviewsRaw = text_dataset_from_directory(IMDB_TEN_FILES_DIR+"/train", batch_size=50, 
                validation_split=None, seed=1337, subset=None, format="grain")
    imdbReviews = next(iter(imdbReviewsRaw))
    return pd.DataFrame({"Generated review":imdbReviews[0],"Expert label":imdbReviews[1]})

def getRefExpertSentiments(reference):
    loadVocabFromFile(SAVE_TO_DIR+"TextClassificationVocab.pkl")
    model=keras.models.load_model(SAVE_TO_DIR+'TextClassificationTorchModel.keras')
    refExpertComments = reference.filter(items=["Expert comment"]).to_numpy().flatten()
    vectorzd_text_batch,labelsVectorized = vectorize_text((refExpertComments,[]))
    evalRes = model(vectorzd_text_batch)
    return evalRes
    
def generateDriftReport(reference, current):
    drift_report = Report([
        DriftedColumnsCount(cat_stattest="psi", num_stattest="wasserstein", per_column_method={"Expert label":"psi"}, drift_share=0.8),
        ValueDrift(column="Generated review", method="perc_text_content_drift"),
        ValueDrift(column="Generated review", method="abs_text_content_drift")],
        ValueDrift(column="Expert label", method="psi", threshold=0.05),     
        ValueDrift(column="Expert label", method="chisquare"),          
        include_tests=False)
    drift_snapshot = drift_report.run(current, reference)
    drift_snapshot.save_html(SAVE_TO_DIR_EVIDENTLY+"/driftSnapshotTextReviews.html")
    
    generator_drift_report = Report([
        ColumnMetricGenerator(ValueDrift, columns=["Generated review"],metric_kwargs={"method":"perc_text_content_drift"}),
        ColumnMetricGenerator(ValueDrift),         
        ColumnMetricGenerator(UniqueValueCount, column_types="cat"),])
    
    generator_drift_snapshot = generator_drift_report.run(current, reference)
    generator_drift_snapshot.save_html(SAVE_TO_DIR_EVIDENTLY+"/genDriftSnapshotTextReviews.html")
 

reference = getRefData()
current = getCurrData()
generateDriftReport(reference,current)