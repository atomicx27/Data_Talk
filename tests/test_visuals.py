"""Tests for expanded complex visuals and stacked bar charts in ChartRecommender."""

import pandas as pd
import pytest
from src.viz.chart_recommender import ChartRecommender


@pytest.fixture
def sample_multidim_df():
    return pd.DataFrame({
        "Department": ["Computer Science", "Electrical", "Mechanical", "Civil", "Computer Science", "Electrical"],
        "Term": ["Fall 2024", "Fall 2024", "Fall 2024", "Fall 2024", "Spring 2025", "Spring 2025"],
        "Enrollments": [450, 320, 280, 210, 480, 310],
        "AverageGPA": [3.65, 3.42, 3.28, 3.15, 3.70, 3.45],
        "TuitionRevenue": [900000.0, 640000.0, 560000.0, 420000.0, 960000.0, 620000.0],
    })


@pytest.fixture
def sample_comparison_df():
    return pd.DataFrame({
        "College": ["Engineering", "Business", "Arts", "Sciences"],
        "PassRate": [92.5, 88.0, 85.2, 90.1],
        "Attendance": [89.0, 93.4, 82.1, 87.5],
        "Satisfaction": [4.5, 4.2, 4.7, 4.1],
    })


def test_chart_recommender_new_types_support():
    for ct in [
        "stacked_bar",
        "stacked_bar_100",
        "stacked_horizontal_bar",
        "donut",
        "heatmap",
        "radar",
        "box",
        "funnel",
        "sunburst",
    ]:
        assert ct in ChartRecommender.SUPPORTED_CHART_TYPES


def test_chart_recommendation_intents(sample_multidim_df):
    assert ChartRecommender.recommend_chart_type(sample_multidim_df, "Show stacked bar of enrollments") == "stacked_bar"
    assert ChartRecommender.recommend_chart_type(sample_multidim_df, "100% stacked bar of students") == "stacked_bar_100"
    assert ChartRecommender.recommend_chart_type(sample_multidim_df, "Stacked horizontal bar of revenue") == "stacked_horizontal_bar"
    assert ChartRecommender.recommend_chart_type(sample_multidim_df, "Donut ring chart of enrollments") == "donut"
    assert ChartRecommender.recommend_chart_type(sample_multidim_df, "Heatmap matrix of department gpa") == "heatmap"
    assert ChartRecommender.recommend_chart_type(sample_multidim_df, "Radar spider comparison") == "radar"
    assert ChartRecommender.recommend_chart_type(sample_multidim_df, "Box plot distribution of scores") == "box"
    assert ChartRecommender.recommend_chart_type(sample_multidim_df, "Conversion funnel pipeline") == "funnel"
    assert ChartRecommender.recommend_chart_type(sample_multidim_df, "Sunburst drill down") == "sunburst"


def test_build_stacked_bar_figures(sample_multidim_df):
    fig_stack = ChartRecommender.build_figure(sample_multidim_df, chart_type="stacked_bar", title="Stacked Bar")
    assert fig_stack is not None
    assert fig_stack.layout.barmode == "stack"

    fig_stack100 = ChartRecommender.build_figure(sample_multidim_df, chart_type="stacked_bar_100", title="100% Stacked")
    assert fig_stack100 is not None
    assert fig_stack100.layout.barnorm == "percent"

    fig_hstacked = ChartRecommender.build_figure(sample_multidim_df, chart_type="stacked_horizontal_bar", title="HStacked")
    assert fig_hstacked is not None
    assert fig_hstacked.layout.barmode == "stack"


def test_build_donut_figure(sample_multidim_df):
    fig = ChartRecommender.build_figure(sample_multidim_df, chart_type="donut", title="Donut")
    assert fig is not None
    assert len(fig.data) == 1
    assert fig.data[0].hole == 0.55


def test_build_heatmap_figure(sample_multidim_df):
    fig = ChartRecommender.build_figure(sample_multidim_df, chart_type="heatmap", title="Cross Tab Heatmap")
    assert fig is not None
    assert len(fig.data) >= 1


def test_build_radar_figure(sample_comparison_df):
    fig = ChartRecommender.build_figure(sample_comparison_df, chart_type="radar", title="Radar Evaluation")
    assert fig is not None
    assert len(fig.data) == len(sample_comparison_df)
    assert fig.data[0].type == "scatterpolar"


def test_build_box_and_funnel_figures(sample_multidim_df):
    fig_box = ChartRecommender.build_figure(sample_multidim_df, chart_type="box", title="Box Plot")
    assert fig_box is not None
    assert fig_box.data[0].type == "box"

    fig_funnel = ChartRecommender.build_figure(sample_multidim_df, chart_type="funnel", title="Funnel Chart")
    assert fig_funnel is not None
    assert fig_funnel.data[0].type == "funnel"


def test_build_sunburst_figure(sample_multidim_df):
    fig_sun = ChartRecommender.build_figure(sample_multidim_df, chart_type="sunburst", title="Sunburst")
    assert fig_sun is not None
    assert fig_sun.data[0].type == "sunburst"
