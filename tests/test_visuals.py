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


def test_gender_ratio_multi_metric_breakdown_recommendation():
    """Verify that multi-metric demographic composition (like gender ratio across colleges)
    recommends a stacked bar and renders all categories (Male, Female, Other) instead of a dual-axis combo.
    """
    df_gender = pd.DataFrame({
        "COLLEGE NAME": ["Engineering", "Business", "Medicine", "Arts", "Science"],
        "MALE COUNT": [450, 320, 210, 180, 290],
        "FEMALE COUNT": [380, 410, 350, 280, 310],
        "OTHER COUNT": [20, 15, 12, 18, 14],
        "TOTAL STUDENTS": [850, 745, 572, 478, 614],
        "MALE PCT": [52.94, 42.95, 36.71, 37.66, 47.23],
        "FEMALE PCT": [44.71, 55.03, 61.19, 58.58, 50.49],
        "OTHER PCT": [2.35, 2.01, 2.10, 3.77, 2.28],
    })

    # Should recommend stacked_bar_100 or stacked_bar, definitely NOT combo
    rec = ChartRecommender.recommend_chart_type(df_gender, "show me gender raio of students for each college")
    assert rec in ("stacked_bar_100", "stacked_bar")
    assert rec != "combo"

    # Figure generation should plot Male, Female, Other traces without double-counting TOTAL STUDENTS
    fig = ChartRecommender.build_figure(df_gender, chart_type=rec, title="Gender Ratio Across Colleges")
    assert fig is not None
    trace_names = [t.name.lower() for t in fig.data]
    assert any("male" in n for n in trace_names)
    assert any("female" in n for n in trace_names)
    assert any("other" in n for n in trace_names)
    assert not any("total" in n for n in trace_names)


def test_high_cardinality_guard_prevents_unreadable_stacked_bar():
    """Verify that high-cardinality entity tables (e.g. 5,000 students) refuse to render an illegible wall of bars."""
    df_large = pd.DataFrame({
        "student_id": [f"STD_{i}" for i in range(1, 1001)],
        "first_name": [f"Name_{i}" for i in range(1, 1001)],
        "gpa": [3.5] * 1000,
        "credits": [60] * 1000,
    })

    # Recommender must fall back to table
    rec = ChartRecommender.recommend_chart_type(df_large, "you are not showing the other students")
    assert rec == "table"

    # build_figure should return None (rendering the interactive table view)
    fig = ChartRecommender.build_figure(df_large, chart_type="stacked_bar", title="Students")
    assert fig is None


def test_long_format_two_dimensions_colored_by_category():
    """Verify that when long-format data has 2 dimensions (e.g. College Name and Gender),
    both stacked_horizontal_bar and horizontal_bar color by the second dimension (Gender)
    and produce distinct category traces/legends rather than monochrome solid blue bars.
    """
    df_long = pd.DataFrame({
        "College Name": ["Architecture", "Architecture", "Architecture", "Business", "Business", "Business"],
        "Gender": ["Female", "Male", "Non-Binary", "Female", "Male", "Non-Binary"],
        "Gender Count": [555, 566, 537, 516, 503, 512],
        "Gender Ratio": [0.335, 0.341, 0.324, 0.337, 0.329, 0.334],
    })

    # Recommendation for breakdown/ratio should be stacked_horizontal_bar or stacked_bar
    rec = ChartRecommender.recommend_chart_type(df_long, "show students gender ritio for each college")
    assert rec in ("stacked_horizontal_bar", "stacked_bar")

    # Building horizontal_bar directly must also color by Gender and have multiple traces/categories
    fig_h = ChartRecommender.build_figure(df_long, chart_type="horizontal_bar", title="Gender by College")
    assert fig_h is not None
    assert len(fig_h.data) == 3
    trace_labels = [t.name for t in fig_h.data]
    assert "Female" in trace_labels and "Male" in trace_labels and "Non-Binary" in trace_labels


def test_constant_percentage_metric_avoids_combo_chart():
    """Verify that if a percentage column is completely constant (e.g. all 100%),
    recommend_chart_type does NOT recommend a misleading dual-axis combo chart with a flat line.
    """
    df_flat = pd.DataFrame({
        "College Name": ["Architecture", "Business", "Education", "Engineering", "Medicine"],
        "Total Faculty": [52, 52, 54, 52, 52],
        "Gender Ratio Pct": [100.0, 100.0, 100.0, 100.0, 100.0],
    })

    rec = ChartRecommender.recommend_chart_type(df_flat, "do same for faculty")
    assert rec != "combo"
    assert rec in ("bar", "horizontal_bar")


def test_automargin_and_transparent_canvas():
    """Verify that chart layout uses transparent canvas background and automargin on both axes."""
    df_sample = pd.DataFrame({
        "Category": ["A", "B", "C"],
        "Value": [10, 20, 30],
    })
    fig = ChartRecommender.build_figure(df_sample, chart_type="bar", title="Sample")
    assert fig is not None
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.xaxis.automargin is True
    assert fig.layout.yaxis.automargin is True


def test_trend_line_chart_plots_fee_due_not_row_count():
    """Verify Screenshot 1 issue: trend query with payment_year, total_fee_due, and row_count
    plots total_fee_due on the Y-axis rather than the row_count metadata.
    """
    df_trend = pd.DataFrame({
        "payment_year": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        "total_fee_due": [6780000.0, 12500000.0, 24000000.0, 45000000.0, 62000000.0, 78000000.0, 89000000.0, 98240000.0],
        "row_count": [1200, 2400, 4800, 8900, 12400, 15600, 17800, 19600],
    })

    rec = ChartRecommender.recommend_chart_type(df_trend, "show me trend of total fees")
    assert rec == "line"

    fig = ChartRecommender.build_figure(df_trend, chart_type=rec, title="Trend of Total Fees")
    assert fig is not None
    assert len(fig.data) >= 1
    # Trace must plot total_fee_due (values in millions), NOT row_count (values in thousands)
    trace = fig.data[0]
    assert trace.y[0] > 1000000.0  # Fees are 6.7M+, row_count is 1200
    assert "total_fee_due" in str(trace.name).lower() or "fees" in str(fig.layout.yaxis.title.text or "").lower() or trace.y[0] == 6780000.0


def test_two_dimension_drill_down_50_rows_recommends_stacked_bar_not_table():
    """Verify Screenshot 2 issue: 2D drill-down with 50 rows (8 colleges x 7 years)
    recommends stacked_horizontal_bar / stacked_bar and renders an interactive figure,
    NOT defaulting to 'table' / None.
    """
    colleges = [
        "College of Architecture & Built Environment",
        "College of Computing & Software",
        "College of Engineering & Technology",
        "College of Arts & Sciences",
        "College of Business",
        "College of Health Sciences",
        "College of Law",
        "College of Education",
    ]
    years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    records = []
    for c in colleges:
        for y in years:
            records.append({
                "college_name": c,
                "payment_year": y,
                "total_fee_due": 1200000.0 * (y - 2018),
                "payment_year_count": 2400,
            })
    df_drill = pd.DataFrame(records[:50])  # 50 rows, exactly as in screenshot
    assert len(df_drill) == 50

    rec = ChartRecommender.recommend_chart_type(df_drill, "Break that down by College")
    assert rec in ("stacked_horizontal_bar", "stacked_bar")
    assert rec != "table"

    fig = ChartRecommender.build_figure(df_drill, chart_type=rec, title="Fees by College and Payment Year")
    assert fig is not None
    assert fig.layout.barmode == "stack"


def test_non_additive_avg_salary_across_years_and_colleges_never_stacks():
    """Verify Power BI / Data Analyst audit fix:
    When breaking down a non-additive measure like avg_salary across College and Hire Year,
    ChartRecommender must NOT stack 19 years of ~$80k averages into a $1.5M bar!
    Instead, it recommends a multi-series 'line' chart with hire_year on the X-axis and college_name as color.
    """
    records = []
    for c in ["College of Business", "College of Engineering", "College of Law"]:
        for y in [2021, 2022, 2023, 2024]:
            records.append({
                "college_name": c,
                "hire_year": y,
                "avg_salary": 85000.0 + (y - 2021) * 2000.0,
                "hire_year_count": 12,
            })
    df_sal = pd.DataFrame(records)

    rec = ChartRecommender.recommend_chart_type(df_sal, "Break that down by College")
    assert rec == "line"

    fig = ChartRecommender.build_figure(df_sal, chart_type=rec, title="Salary Trend by College")
    assert fig is not None
    assert len(fig.data) == 3  # 3 college lines
    # X-axis is hire_year (2021..2024), Y-axis is avg_salary (~85k..91k, NOT $350k+)
    assert list(fig.data[0].x) == [2021, 2022, 2023, 2024]
    assert max(fig.data[0].y) < 120000.0

    # Even if user manually selects stacked_bar on a non-additive metric, barmode must be 'group'
    fig_bar = ChartRecommender.build_figure(df_sal, chart_type="stacked_bar", title="Salary by College")
    assert fig_bar is not None
    assert fig_bar.layout.barmode == "group"





