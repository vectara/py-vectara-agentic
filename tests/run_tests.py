#!/usr/bin/env python3
"""
Custom test runner that suppresses Pydantic deprecation warnings.
Usage: python run_tests.py [test_pattern]
"""

import sys
import warnings
import unittest
import argparse
import asyncio
import gc


def suppress_pydantic_warnings():
    """Comprehensive warning suppression before unittest starts."""
    # Multiple layers of suppression
    warnings.resetwarnings()
    warnings.simplefilter("ignore", DeprecationWarning)

    # Specific Pydantic patterns
    pydantic_patterns = [
        ".*PydanticDeprecatedSince.*",
        ".*__fields__.*deprecated.*",
        ".*__fields_set__.*deprecated.*",
        ".*model_fields.*deprecated.*",
        ".*model_computed_fields.*deprecated.*",
        ".*use.*model_fields.*instead.*",
        ".*use.*model_fields_set.*instead.*",
        ".*Accessing.*model_.*attribute.*deprecated.*",
    ]

    # Resource warning patterns (reduce noise, not critical)
    resource_patterns = [
        ".*unclosed transport.*",
        ".*unclosed <socket\\.socket.*",
        ".*unclosed event loop.*",
        ".*unclosed resource <TCPTransport.*",
        ".*Implicitly cleaning up <TemporaryDirectory.*",
    ]

    for pattern in pydantic_patterns:
        warnings.filterwarnings("ignore", category=DeprecationWarning, message=pattern)

    for pattern in resource_patterns:
        warnings.filterwarnings("ignore", category=ResourceWarning, message=pattern)


def main():
    parser = argparse.ArgumentParser(description="Run tests with warning suppression")
    parser.add_argument(
        "pattern",
        nargs="?",
        default="test_*.py",
        help="Test file pattern (default: test_*.py)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Apply comprehensive warning suppression BEFORE unittest starts
    suppress_pydantic_warnings()

    print(f"🧪 Running tests with pattern: {args.pattern}")
    print("🔇 Pydantic deprecation warnings suppressed")

    # Add tests directory to Python path for relative imports
    import os

    sys.path.insert(0, os.path.abspath("tests"))

    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = "tests"
    suite = loader.discover(start_dir, pattern=args.pattern)

    # Run tests
    verbosity = 2 if args.verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # Cleanup to reduce resource warnings
    try:
        # Close any remaining event loops
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        if loop and not loop.is_closed():
            # Cancel all pending tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()

            # Give tasks a chance to complete cancellation
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )

        # Force garbage collection
        gc.collect()

    except Exception:
        # Don't let cleanup errors affect test results
        pass

    # Exit with proper code
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
