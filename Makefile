all: lint mypy test

lint:
	pylint vectara_agentic || true
	flake8 vectara_agentic tests || true
mypy:
	mypy vectara_agentic || true

test:
	python -m unittest discover -s tests


.PHONY: all lint mypy test
