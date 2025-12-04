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
SAVE_TO_DIR=USER_HOME+"/Tools/models/saved/Evidently/"
IMDB_TEN_FILES_DIR="datasets/aclImdbTen" 

os.nice(10) # Be nice!
os.environ["KERAS_BACKEND"] = "torch"

def getRefData():
    reference=pd.read_csv("datasets/code_review.csv")
    return reference
   
def getCurrData():
    imdbReviewsRaw = text_dataset_from_directory(IMDB_TEN_FILES_DIR+"/train", batch_size=50, 
                validation_split=None, seed=1337, subset=None, format="grain")
    imdbReviews = next(iter(imdbReviewsRaw))
    current=pd.DataFrame({"Generated review":imdbReviews[0]})
    return current
    
reference = getRefData()

current = getCurrData()

drift_report = Report([
    ValueDrift(column="Generated review", method="perc_text_content_drift"),
    ValueDrift(column="Generated review", method="abs_text_content_drift")], 
    include_tests=False)
drift_snapshot = drift_report.run(current, reference)
drift_snapshot.save_html(SAVE_TO_DIR+"/driftSnapshotTextReviews.html")

generator_drift_report = Report([
    ColumnMetricGenerator(ValueDrift, columns=["Generated review"],metric_kwargs={"method":"perc_text_content_drift"}),
    ColumnMetricGenerator(UniqueValueCount, column_types="cat"),])
generator_drift_snapshot = generator_drift_report.run(current, reference)
generator_drift_snapshot.save_html(SAVE_TO_DIR+"/genDriftSnapshotTextReviews.html")
