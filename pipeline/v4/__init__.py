"""Preregistered v4 DEV analysis pipeline.

The package is intentionally independent from the v1-v3 pipeline modules.  Its
public interface is the command line entry point exposed by ``python -m
pipeline.v4``.
"""

from .constants import CACHE_SCHEMA, MOUSE_SCHEMA, RESULT_SCHEMA

__all__ = ["CACHE_SCHEMA", "MOUSE_SCHEMA", "RESULT_SCHEMA"]
