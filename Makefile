.PHONY: help install dev test test-coverage lint format clean run-news run-daily dashboard preload docker-build docker-run ollama-setup check-env

help: ## Show this help message
	@echo "Daily Minutes - Makefile Commands"
	@echo "=================================="
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install core dependencies
	pip install -r requirements.txt

install-minimal: ## Install minimal dependencies (faster)
	pip install -r requirements-minimal.txt

dev: ## Install development dependencies
	pip install -r requirements-dev.txt
	pre-commit install

test: ## Run tests
	pytest tests/ -v

test-coverage: ## Run tests with coverage report
	pytest tests/ --cov=src --cov-report=html --cov-report=term

test-imports: ## Test that all packages import correctly
	python test_imports.py

lint: ## Run linting (flake8, mypy, pylint)
	@echo "Running flake8..."
	-flake8 src tests
	@echo "Running mypy..."
	-mypy src
	@echo "Running pylint..."
	-pylint src

format: ## Format code with black and isort
	black src tests scripts
	isort src tests scripts

clean: ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf data/cache/*
	rm -rf dist build *.egg-info

run-news: ## Run news agent (Phase 1)
	@echo "Running news agent..."
	python scripts/run_news_agent.py

run-daily: ## Run full daily minutes workflow
	@echo "Running daily minutes workflow..."
	python scripts/run_daily_minutes.py

dashboard: ## Launch Streamlit dashboard
	@echo "Starting Streamlit dashboard..."
	streamlit run src/ui/streamlit_app.py --server.port 8501

preload: ## Pre-load all data (news, weather, email, calendar, briefs)
	@echo "Pre-loading all data..."
	./venv/bin/python scripts/preload_all_data.py

docker-build: ## Build Docker image
	docker build -t daily-minutes:latest .

docker-run: ## Run Docker container
	docker run -it --rm -p 8501:8501 daily-minutes:latest

ollama-setup: ## Setup Ollama and pull models
	@echo "Checking Ollama installation..."
	@which ollama > /dev/null || (echo "Installing Ollama..." && curl -fsSL https://ollama.com/install.sh | sh)
	@echo "Pulling required models..."
	ollama pull llama3.2:3b
	@echo "Optional: Pull larger model for better results"
	@echo "  ollama pull mistral:7b"
	@echo "Ollama setup complete!"

ollama-start: ## Start Ollama server
	ollama serve

check-env: ## Check environment setup
	@echo "🔍 Checking Environment"
	@echo "======================="
	@echo ""
	@echo "Python version:"
	@python --version
	@echo ""
	@echo "Pip version:"
	@pip --version
	@echo ""
	@echo "Virtual environment:"
	@which python | grep -q venv && echo "  ✅ Running in venv" || echo "  ❌ Not in venv (run: source venv/bin/activate)"
	@echo ""
	@echo "Ollama:"
	@which ollama > /dev/null && echo "  ✅ Ollama installed" || echo "  ❌ Ollama not found (run: make ollama-setup)"
	@echo ""
	@echo ".env file:"
	@test -f .env && echo "  ✅ .env file exists" || echo "  ❌ .env file missing (copy from .env.example)"
	@echo ""
	@echo "Credentials:"
	@test -d credentials && echo "  ✅ credentials/ directory exists" || echo "  ⚠️  credentials/ directory not found"
	@test -f credentials/gmail_credentials.json && echo "  ✅ Gmail credentials exist" || echo "  ⚠️  Gmail credentials missing (optional for Phase 1)"
	@echo ""
	@echo "Project structure:"
	@test -d src && echo "  ✅ src/ directory exists" || echo "  ❌ src/ directory missing"
	@test -d tests && echo "  ✅ tests/ directory exists" || echo "  ❌ tests/ directory missing"
	@test -d data && echo "  ✅ data/ directory exists" || echo "  ❌ data/ directory missing"
	@echo ""
	@echo "Development tools:"
	@which black > /dev/null && echo "  ✅ black installed" || echo "  ⚠️  black not installed (run: make dev)"
	@which pytest > /dev/null && echo "  ✅ pytest installed" || echo "  ⚠️  pytest not installed (run: make dev)"
	@which pre-commit > /dev/null && echo "  ✅ pre-commit installed" || echo "  ⚠️  pre-commit not installed (run: make dev)"

setup-dirs: ## Create project directory structure
	@echo "Creating project directories..."
	@mkdir -p src/agents
	@mkdir -p src/connectors/mcp
	@mkdir -p src/connectors/news
	@mkdir -p src/connectors/calendar
	@mkdir -p src/connectors/email
	@mkdir -p src/core
	@mkdir -p src/llm
	@mkdir -p src/memory
	@mkdir -p src/orchestration
	@mkdir -p src/observability
	@mkdir -p src/ui/components
	@mkdir -p src/ui/formatters
	@mkdir -p tests/unit
	@mkdir -p tests/integration
	@mkdir -p tests/e2e
	@mkdir -p scripts
	@mkdir -p config/prompts
	@mkdir -p credentials
	@mkdir -p data/cache
	@mkdir -p data/rag_documents
	@mkdir -p data/outputs
	@mkdir -p docs
	@touch src/__init__.py
	@touch tests/__init__.py
	@touch credentials/.gitkeep
	@echo "✅ Project structure created"

init: ## Initialize new project (run once)
	@echo "🚀 Initializing Daily Minutes project..."
	@make setup-dirs
	@test -f .env || cp .env.example .env
	@test -f .gitignore || echo -e "venv/\n__pycache__/\n*.pyc\n.env\ncredentials/*.json\ndata/cache/\ndata/outputs/" > .gitignore
	@echo "✅ Project initialized!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Create virtual environment: python3 -m venv venv"
	@echo "  2. Activate it: source venv/bin/activate"
	@echo "  3. Install dependencies: make install-minimal"
	@echo "  4. Configure .env file"
	@echo "  5. Setup Ollama: make ollama-setup"

quick-check: ## Quick sanity check (imports + env)
	@echo "🧪 Running quick checks..."
	@python test_imports.py || true
	@make check-env
