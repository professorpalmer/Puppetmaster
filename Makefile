.PHONY: test demo doctor adapters clean grok-bot-poc

test:
	python -m unittest discover -s tests -v

demo:
	python -m puppetmaster run "Enterprise workflow" --config examples/enterprise-workflow.json

doctor:
	python -m puppetmaster doctor

adapters:
	python -m puppetmaster adapters

# Local-only Grok Bot remote MCP PoC (loopback + optional cloudflared). Not CI.
grok-bot-poc:
	./scripts/grok-bot-remote-poc.sh

clean:
	rm -rf .puppetmaster .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

