PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
export PYTHONPATH := src

.PHONY: install train test lint format up

install:
	python3 -m venv .venv
	$(PIP) install -U pip
	$(PIP) install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r requirements.txt -r requirements-dev.txt

train:
	$(PY) -m shorts_recsys.train

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests spark
	$(PY) -m ruff format --check src tests spark

format:
	$(PY) -m ruff format src tests spark
	$(PY) -m ruff check --fix src tests spark

up:
	docker compose up --build
