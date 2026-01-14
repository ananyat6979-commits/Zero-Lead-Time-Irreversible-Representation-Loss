.PHONY: test clean

test:
	python -m pytest

clean:
	rm -rf __pycache__ .pytest_cache

