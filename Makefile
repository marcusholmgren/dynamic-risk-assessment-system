.DEFAULT_GOAL := help

help:
		@echo "Please use 'make <target>' where <target> is one of"
		@echo ""
		@echo "  setup       create python virtual environment"
		@echo "  install     install or update python virtual environment"
		@echo "  ingestion   data ingestion"
		@echo ""
		@echo "Check the Makefile to know exactly what each target is doing."

setup:
	python -m venv venv
	source venv/bin/activate

install:
	pip install -r requirements.txt

ingestion:
	python ingestion.py