"""Modular SQL/DAX compilers, executive summary builders, and Chart Thread analysis helpers."""

from src.agents.compilers.deterministic_compiler import (
    build_dax_equivalent,
    clean_agg_alias,
    compile_deterministic_sql,
    infer_agg_function,
    infer_sort_direction,
)
from src.agents.compilers.semantic_helpers import (
    build_adaptive_schema_context,
    check_just_in_time_clarification,
    expand_schema_from_error,
    find_matching_verified_query,
    generate_query,
    improve_user_prompt_payload,
    llm_compile_query,
    resolve_follow_up_context,
    select_adaptive_schema_tiers,
)
from src.agents.compilers.summary_and_followups import (
    build_executive_summary,
    generate_dataset_overview_payload,
    generate_drill_down_options,
    generate_suggested_follow_ups,
    is_dataset_overview_question,
)
from src.agents.compilers.text_matching import (
    detect_categorical_value_matches,
    detect_relevant_tables,
    ensure_categorical_samples,
    extract_what_if_from_prompt,
    fuzzy_matches_word,
    match_measures,
    normalize_prompt_text,
    normalize_what_if_overrides,
)
from src.agents.compilers.thread_analyzer import (
    analyze_chart_dataframe,
    answer_chart_thread_question,
    needs_warehouse_sql_for_thread,
    should_render_chart,
)

__all__ = [
    "analyze_chart_dataframe",
    "answer_chart_thread_question",
    "build_adaptive_schema_context",
    "build_dax_equivalent",
    "build_executive_summary",
    "check_just_in_time_clarification",
    "clean_agg_alias",
    "compile_deterministic_sql",
    "detect_categorical_value_matches",
    "detect_relevant_tables",
    "ensure_categorical_samples",
    "expand_schema_from_error",
    "extract_what_if_from_prompt",
    "find_matching_verified_query",
    "fuzzy_matches_word",
    "generate_dataset_overview_payload",
    "generate_drill_down_options",
    "generate_query",
    "generate_suggested_follow_ups",
    "improve_user_prompt_payload",
    "infer_agg_function",
    "infer_sort_direction",
    "is_dataset_overview_question",
    "llm_compile_query",
    "match_measures",
    "needs_warehouse_sql_for_thread",
    "normalize_prompt_text",
    "normalize_what_if_overrides",
    "resolve_follow_up_context",
    "select_adaptive_schema_tiers",
    "should_render_chart",
]
