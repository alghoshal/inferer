from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval import evaluate
from deepeval.models import OllamaModel
from deepeval.metrics.g_eval import Rubric
from deepeval.metrics import AnswerRelevancyMetric

import os
os.nice(10)  # Be nice!

model = OllamaModel(
    model="llama3.2:1b",
    base_url="http://localhost:11434",
    temperature=0.0,
)

correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    # evaluation_steps=["Check whether the facts are true",],
    evaluation_params=[LLMTestCaseParams.INPUT,
                       LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    model=model,
    rubric=[
        Rubric(score_range=(0, 2), expected_outcome="Factually incorrect."),
        Rubric(score_range=(3, 6), expected_outcome="Mostly correct."),
        Rubric(score_range=(7, 9), expected_outcome="Correct but missing minor details."),
        Rubric(score_range=(10, 10), expected_outcome="100% correct."),
    ],
    #    threshold=0.1
)

answer_relevancy = AnswerRelevancyMetric(threshold=0.5, model=model)

test_case_maths = LLMTestCase(
    input="what is 80 in words? using only 1 word.",
    actual_output="eighty",
    expected_output="eighty"
)

evaluate(test_cases=[test_case_maths], metrics=[answer_relevancy])
