"""Streaming core: event-time records, window assigners, watermarks, engine.

Nothing in this package imports a source, an operator, a sink or scikit-learn.
The core is the part of the system that would survive swapping every adapter.
"""
