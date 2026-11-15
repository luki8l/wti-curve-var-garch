.PHONY: all data analysis assets pdf test lint clean

all: analysis assets pdf

data:
	python scripts/download_data.py

analysis:
	python -m wti.analysis

assets:
	python -m wti.report

pdf:
	cd reports && latexmk -pdf -interaction=nonstopmode report.tex factsheet.tex

test:
	pytest

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

clean:
	cd reports && latexmk -c report.tex factsheet.tex
	rm -rf output .pytest_cache .ruff_cache
