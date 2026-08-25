.PHONY: setup lint type test check cohort run repro smoke docs clean

PY ?= python3
VENV ?= .venv/bin/python

setup:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check wsi_ablation tests

type:
	mypy

test:
	$(PY) -m pytest -q

check: lint type test

cohort:
	$(PY) -m wsi_ablation.cli --config configs/default.yaml cohort

run:
	$(PY) -m wsi_ablation.cli --config configs/default.yaml run

smoke:
	$(PY) -m wsi_ablation.cli --config configs/smoke.yaml run

repro:
	$(PY) -m wsi_ablation.cli --config configs/smoke.yaml repro

docs: run
	$(PY) scripts/build_pages.py

clean:
	rm -rf runs data .pytest_cache .ruff_cache .mypy_cache **/__pycache__ *.egg-info
