FILE := simfempy/__about__.py
.PHONY: clean

all: $(FILE) clean
	python3 setup.py sdist
	twine upload dist/* --verbose
	pip install simfempy --upgrade

clean:
	rm -rf dist build simfempy.egg-info
