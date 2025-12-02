import pandas as pd
import os
from evidently.metrics import DriftedColumnsCount
from evidently.metrics import ValueDrift
from evidently import Report
from evidently.generators import ColumnMetricGenerator
from evidently.metrics import UniqueValueCount

USER_HOME=os.path.expanduser("~")
SAVE_TO_DIR=USER_HOME+"/Tools/models/saved/Evidently/"

os.nice(10) # Be nice!

reference_data = {
        "Question": [
            "How can I manage stress effectively?",
            "What is the best way to treat a minor burn?",
            "How often should I get a medical check-up?",
            "What foods should I eat to boost my immune system?",
            "What medication should I take for depression without consulting a doctor?"
        ],
        "Answer": [
            "Manage stress by practicing relaxation techniques like deep breathing, exercise, and mindfulness meditation. Check out this link http://localhost.com",
            "For a minor burn, cool the area with running water for 10 minutes and cover it with a sterile dressing.",
            "I'm sorry, I cannot provide specific advice on this question. Please consult a healthcare professional.",
            "I'm sorry, I cannot answer this question. Please seek advice from a licensed healthcare provider.",
            "You could try taking something over-the-counter, like aspirin or ibuprofen, for depression."
        ],
        "Rating":
        [
            5,
            4,
            0,
            1,
            0
        ],
        "Feedback": [
            "Positive",
            "Positive",
            "Negative",
            "Negative",
            "Negative"
        ],
        "Predicted Feedback": [
            "Positive",
            "Negative",
            "Negative",
            "Negative",
            "Positive"
        ],
        "Predicted Probas": [
            0.85,
            0.9,
            0.1,
            0.15,
            0.6, 
        ],
    }
reference = pd.DataFrame(reference_data)

current_data = {
      "Question": [
          "What are some effective ways to improve sleep quality?",
          "How can I safely exercise with a knee injury?",
          "Is it safe to take vitamins every day?",
          "What are natural remedies for headaches?",
          "Can I stop taking antibiotics if I feel better?"
     ],
      "Answer": [
          "To improve sleep, maintain a regular sleep schedule, avoid screens before bed, and create a relaxing bedtime routine.",
          "Consult a physiotherapist for exercises that don't strain your knee. Low-impact activities like swimming may help.",
          "I'm sorry, I cannot provide medical advice. Please consult a healthcare provider.",
          "I'm sorry, but I can't give medical advice. Please consult a healthcare professional.",
          "It’s important to complete the full course of antibiotics as prescribed, even if you feel better."
     ],
      "Rating": [
         5,
         4,
         0,
         1,
         3
     ],
      "Feedback": [
          "Positive",
          "Positive",
          "Negative",
          "Negative",
          "Negative"
     ],
      "Predicted Feedback": [
          "Positive",
          "Negative",
          "Negative",
          "Negative",
          "Positive"
     ],
      "Predicted Probas": [
         0.9,
         0.75,
         0.2,
         0.1,
         0.65
     ],
}
current = pd.DataFrame(current_data)

drift_report = Report([
    DriftedColumnsCount(cat_stattest="psi", num_stattest="wasserstein", per_column_method={"Feedback":"psi", "Predicted Feedback":"psi"}, drift_share=0.8),
    ValueDrift(column="Feedback", method="psi", threshold=0.05),
    ValueDrift(column="Rating", method="chisquare"),
    ValueDrift(column="Question", method="perc_text_content_drift"),
    ValueDrift(column="Answer", method="abs_text_content_drift")], 
    include_tests=False)
drift_snapshot = drift_report.run(current, reference)
drift_snapshot.save_html(SAVE_TO_DIR+"/driftSnapshot.html")

generator_drift_report = Report([ColumnMetricGenerator(ValueDrift, columns=["Question", "Answer"],
        metric_kwargs={"method":"perc_text_content_drift"}),
        ColumnMetricGenerator(ValueDrift), 
        ColumnMetricGenerator(UniqueValueCount, column_types="cat"),])
generator_drift_snapshot = generator_drift_report.run(current, reference)
generator_drift_snapshot.save_html(SAVE_TO_DIR+"/genDriftSnapshot.html")