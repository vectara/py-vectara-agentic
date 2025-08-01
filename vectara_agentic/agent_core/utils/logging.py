"""
Logging configuration and utilities for agent functionality.

This module provides logging filters, configuration, and setup utilities
specifically tailored for agent operations and debugging.
"""

import logging
from dotenv import load_dotenv


class IgnoreUnpickleableAttributeFilter(logging.Filter):
    """
    Filter to ignore log messages that contain certain strings.

    This filter is used to suppress common unpickleable attribute warnings
    that occur during agent serialization/deserialization operations.
    """

    def filter(self, record):
        """
        Filter log records based on message content.

        Args:
            record: LogRecord to evaluate

        Returns:
            bool: True if record should be logged, False if it should be ignored
        """
        msgs_to_ignore = [
            "Removing unpickleable private attribute _split_fns",
            "Removing unpickleable private attribute _sub_sentence_split_fns",
        ]
        return all(msg not in record.getMessage() for msg in msgs_to_ignore)


def setup_agent_logging():
    """
    Set up logging configuration for agent operations.

    This configures logging filters and levels to reduce noise from
    agent-related operations while maintaining useful debug information.
    """
    # Add filter to suppress unpickleable attribute warnings
    logging.getLogger().addFilter(IgnoreUnpickleableAttributeFilter())

    # Set critical level for OTLP trace exporter to reduce noise
    logger = logging.getLogger("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    logger.setLevel(logging.CRITICAL)

    # Load environment variables with override
    load_dotenv(override=True)
