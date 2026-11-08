import re
import pandas as pd
import pytest

from src.agents.query_agent import QueryAgent


def test_ordinal_and_typo_thread_analysis():
    data = [
        {"college_name": "College of Humanities & Social Sciences", "gender": "Non-Binary", "gender_ratio_percent": 34.87},
        {"college_name": "College of Law & Public Policy", "gender": "Non-Binary", "gender_ratio_percent": 34.50},
        {"college_name": "College of Architecture & Built Environment", "gender": "Male", "gender_ratio_percent": 34.10},
        {"college_name": "College of Engineering & Computing", "gender": "Non-Binary", "gender_ratio_percent": 34.00},
        {"college_name": "College of Education & Human Development", "gender": "Female", "gender_ratio_percent": 31.87},
    ]
    df = pd.DataFrame(data)

    # Q1: "what college has the hight ratio" (typo 'hight' for highest)
    ans1, _, _ = QueryAgent._analyze_chart_dataframe("what college has the hight ratio", df)
    assert "College of Humanities & Social Sciences" in ans1
    assert "Non-Binary" in ans1
    assert "34.87" in ans1

    # Q2: "2nd hights ratio" (ordinal rank 2 + typo 'hights' for highest)
    ans2, _, _ = QueryAgent._analyze_chart_dataframe("2nd hights ratio", df)
    assert "College of Law & Public Policy" in ans2
    assert "Non-Binary" in ans2
    assert "34.50" in ans2
    assert "2nd" in ans2


def test_primary_llm_answering_in_chart_thread():
    """Verify that when an LLM is configured, it is called FIRST to answer directly."""
    data = [
        {"college_name": "College of Humanities & Social Sciences", "gender": "Non-Binary", "gender_ratio_percent": 34.87},
        {"college_name": "College of Law & Public Policy", "gender": "Non-Binary", "gender_ratio_percent": 34.50},
    ]
    df = pd.DataFrame(data)

    class MockLLMRouter:
        def is_configured(self):
            return True

        def complete(self, purpose, sys_prompt, usr_prompt):
            assert "what college has the hight ratio" in usr_prompt or "2nd hights" in usr_prompt
            return "The College of Humanities & Social Sciences (Non-Binary) has the highest ratio at 34.87%."

    mock_llm = MockLLMRouter()
    ans, _, _ = QueryAgent._analyze_chart_dataframe(
        "what college has the hight ratio",
        df,
        parent_question="gender ratio by college",
        llm_router=mock_llm,
    )
    assert ans == "The College of Humanities & Social Sciences (Non-Binary) has the highest ratio at 34.87%."

