.DEFAULT_GOAL := help

help:
		@echo "Please use 'make <target>' where <target> is one of"
		@echo ""
		@echo "  setup       create python virtual environment"
		@echo "  install     install or update python virtual environment"
		@echo "  ingestion   1. data ingestion"
		@echo "  training    2. model training"
		@echo "  scoring     3. model scoring"
		@echo "  deploy      4. model deployment"
		@echo ""
		@echo "Check the Makefile to know exactly what each target is doing."

setup:
	python -m venv venv
	source venv/bin/activate

install:
	pip install -r requirements.txt

ingestion:
	python ingestion.py

training:
	python training.py

scoring:
	python scoring.py

deploy:
	python deployment.py
