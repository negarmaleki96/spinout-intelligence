# Spinout Intelligence MVP

Paper-first vertical slice: OpenAlex paper discovery, Crossref DOI enrichment, Papers with Code search, GitHub repository discovery, and SQLite persistence.

## Setup
Python 3.11+
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Optional credentials: `OPENALEX_API_KEY` (free key recommended for larger paging workloads), `GITHUB_TOKEN` (higher rate limit), and `CROSSREF_MAILTO`.

## Run
```bash
spinout discover --topic robotics --from-date 2022-01-01 --to-date 2026-10-24 --max-papers 100
spinout report
spinout report --format csv --output candidates.csv
pytest
```

All paper-to-repository matches are review candidates, not verified links. GitHub stars/forks and model downloads are not customers, revenue, or proof of commercialization. Prototype only.
