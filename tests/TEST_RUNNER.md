# Custom Test Runner

This project includes a custom test runner (`run_tests.py`) that suppresses external dependency warnings, particularly Pydantic deprecation warnings that appear during test execution.

## Usage

### Run all tests:
```bash
python tests/run_tests.py
```

### Run specific test file:
```bash
python tests/run_tests.py test_agent.py
```

### Run with verbose output:
```bash
python tests/run_tests.py --verbose
```

### Run tests matching a pattern:
```bash
python tests/run_tests.py "test_agent*"
```

## Standard unittest Alternative

If you prefer using standard unittest and don't mind the warnings, you can still run:
```bash
python -m unittest discover -s tests -p "test_*.py"
```
