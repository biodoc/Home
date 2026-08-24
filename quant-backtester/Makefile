# Convenience wrappers. Everything here is a one-line command you can also run
# by hand -- nothing is hidden behind the Makefile.
#
# On Windows, run the underlying commands directly (see README) or use `make`
# from WSL / Git Bash.

PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin
TICKERS ?= UAMY SMR BBAI AEM
SIGNALS ?= all

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)

.PHONY: setup
setup: $(BIN)/python  ## Create the venv and install everything
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -r requirements.txt
	$(BIN)/python -m pip install -e .
	@echo
	@$(BIN)/python backtest.py --doctor || true

.PHONY: doctor
doctor:  ## Check this machine is set up correctly
	$(BIN)/python backtest.py --doctor

.PHONY: test
test:  ## Run the test suite (no network needed)
	$(BIN)/python -m pytest

.PHONY: run
run:  ## Backtest TICKERS with SIGNALS (override on the command line)
	$(BIN)/python backtest.py --tickers $(TICKERS) --signals $(SIGNALS)

.PHONY: full
full:  ## Full run: all signals, diagnostics, hold-out validation, CSV out
	$(BIN)/python backtest.py --signals all --diagnostics --validate \
		--out results/latest.csv

.PHONY: signals
signals:  ## Describe the available signals
	$(BIN)/python backtest.py --list-signals

.PHONY: clean
clean:  ## Remove caches and build artifacts (keeps data_cache)
	rm -rf .pytest_cache **/__pycache__ qsb/__pycache__ tests/__pycache__ \
		build dist *.egg-info
	find . -name '*.pyc' -delete

.PHONY: clean-data
clean-data:  ## Delete cached market data (forces a re-download)
	rm -rf data_cache
