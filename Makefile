.PHONY: install lint format test run-api run-dashboard clean

PYTHON = venv/bin/python
PIP = venv/bin/pip
PYTEST = venv/bin/pytest
RUFF = venv/bin/ruff
MYPY = venv/bin/mypy

install:
	python3 -m venv venv
	$(PIP) install --upgrade pip
	$(PIP) install -e .
	$(PIP) install ruff mypy

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

test:
	export PYTHONPATH=$$PYTHONPATH:. && $(PYTEST)

run-api:
	$(PYTHON) -m uvicorn api.main:app --reload

run-dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache
