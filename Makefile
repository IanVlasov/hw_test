mypy: ## Run mypy
	mypy \
		--install-types --non-interactive \
		--ignore-missing-imports \
		--follow-imports=silent \
		--show-column-numbers \
		--python-version 3.9 \
		.
