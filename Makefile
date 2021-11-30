FILE := setup.py
.PHONY: clean

all: $(FILE)
	rm -rf dist build simfempy.egg-info
	python3 setup.py sdist
	twine upload dist/* --verbose
	pip install --upgrade simfempy
