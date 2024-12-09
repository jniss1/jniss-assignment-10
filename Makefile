# Makefile for flask app setup
PYTHON := python3.10

install:
	@echo "Using $(PYTHON)"
	$(PYTHON) -m pip install -r requirements.txt
run:
	flask run --port 3000