"""
narremgen
=========
A modular framework for narrative and emotional text generation based on
structured narrative schemes (SN) and emotional dynamics (DE).

This package provides:
- Controlled generation of micro-narratives using OpenAI models.
- Data pipelines for Advice, Mapping, and Context CSVs.
- Narrative composition and batch orchestration.
- Statistical and correspondence analyses of SN/DE distributions.
- Optional graphical interface for manual segmentation.

Core papers:
- Priam, R. (2025). *Narrative and Emotional Structures for Generation of Short Texts for Advice* (Hal-Inria).
- Priam, R. (2025). *Unified Narrative Grammar and Algebra* (unpublished).

Example:
---------
    from narremgen.pipeline import run_pipeline
    run_pipeline("Urban_Walk", assets_dir="settings", output_dir="outputs")
"""

__version__ = "0.9.0"
__author__ = "R. Priam"
__license__ = "MIT"
__email__ = "rpriam@gmail.com"

from .core import (
    get_openai_key,
    safe_chat_completion,
    generate_text,
    estimate_tokens,
)

from .data import (
    generate_advice,
    generate_mapping,
    generate_context,
)

from .narratives import (
    generate_narratives,
    generate_narratives_batch,
)

from .pipeline import run_pipeline

from .analyzestats import (
    analyze_sn_de_distribution,
)

from .utils import (
    save_output,
    safe_generate,
    merge_and_filter,
    renumerote_filtered,
    audit_filtered,
    validate_mapping,
    quick_check_filtered,
)

__all__ = [
    "get_openai_key",
    "safe_chat_completion",
    "generate_text",
    "estimate_tokens",
    "generate_advice",
    "generate_mapping",
    "generate_context",
    "generate_narratives",
    "generate_narratives_batch",
    "run_pipeline",
    "analyze_sn_de_distribution",
    "save_output",
    "safe_generate",
    "merge_and_filter",
    "renumerote_filtered",
    "audit_filtered",
    "validate_mapping",
    "quick_check_filtered",
]

