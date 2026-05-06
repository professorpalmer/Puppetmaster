.PHONY: test demo doctor adapters clean

test:
	python -m unittest discover -s tests -v

demo:
	python -m puppetmaster run "Enterprise workflow" --config examples/enterprise-workflow.json

doctor:
	python -m puppetmaster doctor

adapters:
	python -m puppetmaster adapters

clean:
	rm -rf .puppetmaster .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

