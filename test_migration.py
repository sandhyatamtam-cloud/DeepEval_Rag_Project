"""Create the RAG Test Case"""
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval import assert_test
from github_judge import GitHubModelJudge

gh_judge = GitHubModelJudge()

def test_migration_rationalization():
    query = "Should I migrate my legacy Java app to AWS Lambda?"
    #actual_output = "Yes, AWS Lambda is great for legacy Java because it's serverless."
    actual_output = (
        "Migrating a legacy Java app to AWS Lambda requires caution. "
        "Such applications often face cold start issues and need refactoring "
        "to run efficiently in a serverless environment."
    )


    retrieval_context = [
        "Legacy Java apps with large memory footprints often suffer from cold starts on Lambda.",
        "Refactoring is required for Java apps to run efficiently in serverless environments."
    ]

    faithfulness = FaithfulnessMetric(threshold=0.7, model=gh_judge)
    relevancy = AnswerRelevancyMetric(threshold=0.7, model=gh_judge)

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )

    assert_test(test_case, [faithfulness, relevancy])
