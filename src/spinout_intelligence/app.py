import base64
import csv
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from typing import Any

import httpx
import typer
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.table import Table
from sqlalchemy import create_engine, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Settings(BaseSettings):
    openalex_api_key: str | None = None
    github_token: str | None = None
    crossref_mailto: str | None = None
    database_url: str = "sqlite:///spinout_intelligence.db"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


cfg = Settings()


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    doi: Mapped[str | None] = mapped_column(String(500))
    title: Mapped[str] = mapped_column(Text)
    publication_date: Mapped[str | None] = mapped_column(String(32))
    cited_by_count: Mapped[int | None] = mapped_column(Integer)
    abstract: Mapped[str | None] = mapped_column(Text)


class Researcher(Base):
    __tablename__ = "researchers"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    display_name: Mapped[str] = mapped_column(Text)
    orcid: Mapped[str | None] = mapped_column(String(255))


class Institution(Base):
    __tablename__ = "institutions"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    display_name: Mapped[str] = mapped_column(Text)
    ror: Mapped[str | None] = mapped_column(String(255))
    country_code: Mapped[str | None] = mapped_column(String(8))
    institution_type: Mapped[str | None] = mapped_column(String(100))


class Authorship(Base):
    __tablename__ = "paper_authorships"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paper_id: Mapped[str] = mapped_column(String(255), index=True)
    researcher_id: Mapped[str] = mapped_column(String(255), index=True)
    author_position: Mapped[str | None] = mapped_column(String(32))


class Repo(Base):
    __tablename__ = "repositories"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    github_id: Mapped[int] = mapped_column(Integer, unique=True)
    full_name: Mapped[str] = mapped_column(String(500))
    url: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(String(100))
    stars: Mapped[int] = mapped_column(Integer, default=0)
    forks: Mapped[int] = mapped_column(Integer, default=0)
    pushed_at: Mapped[str | None] = mapped_column(String(64))


class PaperRepo(Base):
    __tablename__ = "paper_repositories"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paper_id: Mapped[str] = mapped_column(String(255), index=True)
    repo_id: Mapped[int] = mapped_column(Integer, index=True)
    match_method: Mapped[str] = mapped_column(String(100))
    confidence: Mapped[float] = mapped_column(Float)
    review_status: Mapped[str] = mapped_column(String(32), default="needs_review")


class RawRecord(Base):
    __tablename__ = "raw_records"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(50))
    external_id: Mapped[str] = mapped_column(String(1000))
    payload_hash: Mapped[str] = mapped_column(String(64))
    payload: Mapped[str] = mapped_column(Text)


class RepoAnalysis(Base):
    __tablename__ = "repository_analyses"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paper_id: Mapped[str] = mapped_column(String(255), index=True)
    repo_id: Mapped[int] = mapped_column(Integer, index=True)
    research_connection_score: Mapped[float] = mapped_column(Float, default=0.0)
    implementation_maturity_score: Mapped[float] = mapped_column(Float, default=0.0)
    classification: Mapped[str] = mapped_column(String(64), default="unclear")
    evidence_json: Mapped[str] = mapped_column(Text, default="[]")
    analyzed_at: Mapped[str] = mapped_column(String(64))


# Metadata labels are provisional; content analysis is limited to README and root-level paths.
def normalize_text(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def classify_repository(full_name: str, description: str | None) -> str:
    text = normalize_text(f"{full_name} {description or ''}")
    if any(term in text for term in ("technical disclosure", "invention disclosure", "technology disclosure", "patent")):
        return "technical_disclosure_candidate"
    if any(term in text for term in ("dataset", "benchmark dataset", "corpus")):
        return "dataset_candidate"
    if any(term in text for term in ("implementation", "reproduction", "experimental code", "research code", "research prototype")):
        return "research_software_candidate"
    if any(term in text for term in ("framework", "library", "toolkit", "application")):
        return "general_software_candidate"
    return "unclear_metadata"


def score_paper_repository(paper: Paper, repo: Repo) -> tuple[float, str]:
    """Conservative metadata score; this is not a calibrated probability.

    A title or DOI appearing in a repository name/description is only
    association evidence. It does not establish that the repository contains
    the paper's implementation.
    """
    title = normalize_text(paper.title)
    doi = normalize_text(paper.doi)
    repo_name = normalize_text(repo.full_name)
    description = normalize_text(repo.description)
    repo_metadata = f"{repo_name} {description}".strip()

    # Metadata references are useful leads, but are not proof of implementation.
    if doi and doi in repo_metadata:
        return 0.50, "doi_in_repository_metadata"

    if title and title in repo_name:
        return 0.35, "exact_title_in_repository_name"

    if title and title in description:
        return 0.45, "exact_title_in_repository_description"

    stopwords = {
        "the", "and", "for", "with", "from", "using", "based",
        "study", "analysis", "research", "new", "of", "in", "on",
        "a", "an", "to",
    }
    title_tokens = set(title.split()) - stopwords
    metadata_tokens = set(repo_metadata.split()) - stopwords

    if not title_tokens:
        return 0.0, "insufficient_title_tokens"

    overlap = len(title_tokens & metadata_tokens) / len(title_tokens)
    if overlap >= 0.70:
        return 0.40, "strong_title_token_overlap"
    if overlap >= 0.45:
        return 0.30, "moderate_title_token_overlap"
    if overlap >= 0.25:
        return 0.20, "weak_title_token_overlap"
    return 0.10, "no_meaningful_metadata_match"


def fetch_repository_contents(github: httpx.Client, full_name: str) -> tuple[str, list[str]]:
    """Fetch README text and root-level paths using GitHub's public REST API."""
    readme_text = ""
    paths: list[str] = []
    try:
        response = github.get(f"/repos/{full_name}/readme")
        if response.status_code == 200:
            readme_data = response.json()
            encoded = readme_data.get("content", "")
            if encoded:
                readme_text = base64.b64decode(encoded).decode("utf-8", errors="replace")
    except (httpx.HTTPError, ValueError, KeyError):
        pass

    try:
        response = github.get(f"/repos/{full_name}/contents")
        if response.status_code == 200 and isinstance(response.json(), list):
            paths = [item.get("name", "") for item in response.json() if item.get("name")]
    except (httpx.HTTPError, ValueError):
        pass
    return readme_text[:100_000], paths


def analyze_repository(paper: Paper, repo: Repo, readme: str, paths: list[str]) -> dict:
    """Assess paper connection and implementation evidence separately.

    Scores are transparent heuristics, not calibrated probabilities. Inspection
    is currently limited to README text and root-level GitHub paths.
    """
    readme_body = "\n".join(
        line for line in readme.splitlines()
        if not line.lstrip().startswith("#")
    )
    readme_norm = normalize_text(readme_body)
    title_norm = normalize_text(paper.title)
    doi_norm = normalize_text(paper.doi)
    repo_name_norm = normalize_text(repo.full_name)
    description_norm = normalize_text(repo.description)

    evidence: list[str] = []
    connection = 0.05  # no demonstrated connection; retain a small neutral floor

    # Strongest evidence: the normalized DOI occurs in README body text.
    if doi_norm and doi_norm in readme_norm:
        connection = 0.98
        evidence.append("DOI explicitly referenced in README body")
    # A paper title in prose is useful, but can still be copied or coincidental.
    elif title_norm and title_norm in readme_norm:
        connection = 0.55
        evidence.append("Exact paper title mentioned in README body; verify authorship/context")
    elif title_norm and title_norm in description_norm:
        connection = 0.40
        evidence.append("Exact paper title mentioned in repository description")
    elif title_norm and title_norm in repo_name_norm:
        connection = 0.30
        evidence.append("Paper title appears in repository name only")
    else:
        stopwords = {
            "the", "and", "for", "with", "from", "using", "based", "study",
            "analysis", "research", "new", "of", "in", "on", "a", "an", "to",
        }
        title_tokens = set(title_norm.split()) - stopwords
        readme_tokens = set(readme_norm.split())
        if title_tokens:
            overlap = len(title_tokens & readme_tokens) / len(title_tokens)
            if overlap >= 0.70:
                connection = 0.20
                evidence.append("High title-token overlap only; topical lead, not a confirmed paper link")
            elif overlap >= 0.40:
                connection = 0.12
                evidence.append("Partial title-token overlap only; weak topical lead")

    path_text = " ".join(paths).lower()
    readme_lower = readme.lower()
    code_extensions = (".py", ".ipynb", ".r", ".jl", ".java", ".cpp", ".c", ".js", ".ts", ".go", ".rs")
    code_files = [name for name in paths if name.lower().endswith(code_extensions)]
    notebooks = [name for name in paths if name.lower().endswith(".ipynb")]
    dataset_terms = ("dataset", "datasets", "benchmark", "corpus")
    doc_terms = ("technical disclosure", "invention disclosure", "patent", "technology disclosure")
    config_names = {"pyproject.toml", "requirements.txt", "environment.yml", "package.json", "cargo.toml"}

    maturity = 0.05
    if code_files:
        maturity = max(maturity, 0.35)
        evidence.append(f"Source-code files found at repository root ({len(code_files)})")
    if notebooks:
        maturity = max(maturity, 0.50)
        evidence.append(f"Notebook files found at repository root ({len(notebooks)})")
    has_dataset_indicator = any(term in path_text or term in readme_lower for term in dataset_terms)
    if has_dataset_indicator:
        maturity = max(maturity, 0.30)
        evidence.append("Dataset or benchmark indicators found")
    if any(name.lower() in config_names for name in paths):
        maturity = max(maturity, 0.60)
        evidence.append("Dependency or project configuration file found")
    if readme.strip():
        evidence.append("README available for inspection")

    if any(term in readme_lower for term in doc_terms):
        classification = "technical_disclosure_candidate"
    elif notebooks and code_files:
        classification = "research_prototype_candidate"
    elif code_files:
        classification = "software_implementation_candidate"
    elif has_dataset_indicator:
        classification = "dataset_candidate"
    elif readme.strip():
        classification = "documentation_or_unclear"
    else:
        classification = "unclear"

    return {
        "research_connection_score": round(connection, 2),
        "implementation_maturity_score": round(maturity, 2),
        "classification": classification,
        "evidence": evidence,
    }


engine = create_engine(
    cfg.database_url,
    connect_args={"check_same_thread": False} if cfg.database_url.startswith("sqlite") else {},
)
Session = sessionmaker(bind=engine)
console = Console()
cli = typer.Typer(no_args_is_help=True)


def client(base: str, token: str | None = None) -> httpx.Client:
    headers = {"User-Agent": "SpinoutIntelligence/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(base_url=base, headers=headers, timeout=30, follow_redirects=True)


def get(c: httpx.Client, path: str, params: dict | None = None) -> dict[str, Any]:
    for attempt in range(4):
        response = c.get(path, params=params)
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        return response.json()
    response.raise_for_status()
    return {}


def norm_doi(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().lower()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value)
    return value.removeprefix("doi:").strip()


def abstract_text(index: dict | None) -> str | None:
    if not index:
        return None
    words = {position: word for word, positions in index.items() for position in positions}
    return " ".join(words[position] for position in sorted(words))


def raw(session, source: str, external_id: str | int, obj: dict) -> None:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    # Avoid storing the same source payload repeatedly on every discovery run.
    exists = session.query(RawRecord).filter_by(
        source=source, external_id=str(external_id), payload_hash=payload_hash
    ).first()
    if not exists:
        session.add(RawRecord(source=source, external_id=str(external_id),
                              payload_hash=payload_hash, payload=payload))


@cli.command()
def discover(
    topic: str = typer.Option(..., help="Research topic to search in OpenAlex"),
    from_date: str = typer.Option("2022-01-01", help="Start date (YYYY-MM-DD)"),
    to_date: str = typer.Option("2026-10-24", help="End date (YYYY-MM-DD)"),
    max_papers: int = typer.Option(100, min=1),
    max_repos: int = typer.Option(3, min=1),
):
    """Discover papers, enrich DOI records, and find candidate GitHub repositories."""
    Base.metadata.create_all(engine)
    oa = client("https://api.openalex.org")
    cr = client("https://api.crossref.org")
    gh = client("https://api.github.com", cfg.github_token)
    pwc = client("https://paperswithcode.co/api/v1")
    cursor, processed = "*", 0

    try:
        with Session() as session:
            while processed < max_papers:
                params = {
                    "search": topic,
                    "filter": f"from_publication_date:{from_date},to_publication_date:{to_date}",
                    "per-page": min(100, max_papers - processed),
                    "cursor": cursor,
                    "select": "id,doi,title,publication_date,authorships,cited_by_count,abstract_inverted_index",
                }
                if cfg.openalex_api_key:
                    params["api_key"] = cfg.openalex_api_key
                batch = get(oa, "/works", params)
                works = batch.get("results", [])
                if not works:
                    break

                for work in works:
                    processed += 1
                    paper_id = work["id"]
                    doi = norm_doi(work.get("doi"))
                    abstract = abstract_text(work.get("abstract_inverted_index"))

                    if doi:
                        try:
                            crossref_params = {"mailto": cfg.crossref_mailto} if cfg.crossref_mailto else None
                            crossref_work = get(cr, f"/works/{doi}", crossref_params).get("message", {})
                            raw(session, "crossref", doi, crossref_work)
                            abstract = abstract or crossref_work.get("abstract")
                        except httpx.HTTPError:
                            pass

                    paper = session.get(Paper, paper_id)
                    if paper is None:
                        paper = Paper(
                            id=paper_id, doi=doi, title=work.get("title") or "Untitled",
                            publication_date=work.get("publication_date"),
                            cited_by_count=work.get("cited_by_count"), abstract=abstract,
                        )
                        session.add(paper)
                    else:
                        paper.doi = doi or paper.doi
                        paper.title = work.get("title") or paper.title
                        paper.publication_date = work.get("publication_date") or paper.publication_date
                        paper.cited_by_count = work.get("cited_by_count")
                        paper.abstract = abstract or paper.abstract

                    raw(session, "openalex", paper_id, work)

                    for authorship in work.get("authorships", []):
                        author = authorship.get("author") or {}
                        author_id = author.get("id")
                        if author_id:
                            if session.get(Researcher, author_id) is None:
                                session.add(Researcher(
                                    id=author_id,
                                    display_name=author.get("display_name") or "Unknown",
                                    orcid=author.get("orcid"),
                                ))
                            session.flush()
                            if not session.query(Authorship).filter_by(
                                paper_id=paper_id, researcher_id=author_id
                            ).first():
                                session.add(Authorship(
                                    paper_id=paper_id, researcher_id=author_id,
                                    author_position=authorship.get("author_position"),
                                ))
                        for institution in authorship.get("institutions", []):
                            institution_id = institution.get("id")
                            if institution_id and session.get(Institution, institution_id) is None:
                                session.add(Institution(
                                    id=institution_id,
                                    display_name=institution.get("display_name") or "Unknown",
                                    ror=institution.get("ror"),
                                    country_code=institution.get("country_code"),
                                    institution_type=institution.get("type"),
                                ))

                    # Candidate discovery: search GitHub by exact title/DOI,
                    # and also collect GitHub links exposed by Papers with Code.
                    candidates = []
                    queries = [f'"{work.get("title", "")}"']
                    if doi:
                        queries.append(f'"{doi}"')
                    for query in queries:
                        try:
                            result = get(gh, "/search/repositories", {"q": query, "per_page": max_repos})
                            candidates.extend(result.get("items", []))
                        except httpx.HTTPError:
                            pass

                    try:
                        pwc_result = get(pwc, "/papers/search", {
                            "q": work.get("title", ""), "mode": "keyword", "page_size": 20,
                        })
                        pwc_items = pwc_result if isinstance(pwc_result, list) else pwc_result.get(
                            "results", pwc_result.get("papers", [])
                        )
                        for item in pwc_items:
                            for url in re.findall(
                                r"https?://github\.com/[\w.-]+/[\w.-]+", json.dumps(item)
                            ):
                                candidates.append({"full_name": url.split("github.com/")[-1].rstrip("/")})
                    except (httpx.HTTPError, AttributeError):
                        pass

                    unique_candidates = {
                        item.get("full_name"): item for item in candidates
                        if item.get("full_name") and item["full_name"].count("/") == 1
                    }

                    for repo_name in list(unique_candidates)[:max_repos]:
                        try:
                            repo_data = get(gh, f"/repos/{repo_name}")
                        except httpx.HTTPError:
                            continue

                        repo = session.query(Repo).filter_by(github_id=repo_data["id"]).first()
                        if repo is None:
                            repo = Repo(
                                github_id=repo_data["id"], full_name=repo_data["full_name"],
                                url=repo_data["html_url"], description=repo_data.get("description"),
                                language=repo_data.get("language"),
                                stars=repo_data.get("stargazers_count", 0),
                                forks=repo_data.get("forks_count", 0),
                                pushed_at=repo_data.get("pushed_at"),
                            )
                            session.add(repo)
                        else:
                            repo.full_name = repo_data["full_name"]
                            repo.url = repo_data["html_url"]
                            repo.description = repo_data.get("description")
                            repo.language = repo_data.get("language")
                            repo.stars = repo_data.get("stargazers_count", 0)
                            repo.forks = repo_data.get("forks_count", 0)
                            repo.pushed_at = repo_data.get("pushed_at")
                        session.flush()

                        confidence, evidence = score_paper_repository(paper, repo)
                        repo_type = classify_repository(repo.full_name, repo.description)
                        readme, root_paths = fetch_repository_contents(gh, repo.full_name)
                        analysis = analyze_repository(paper, repo, readme, root_paths)
                        repo_analysis = session.query(RepoAnalysis).filter_by(
                            paper_id=paper_id, repo_id=repo.id
                        ).first()
                        if repo_analysis is None:
                            repo_analysis = RepoAnalysis(paper_id=paper_id, repo_id=repo.id)
                            session.add(repo_analysis)
                        repo_analysis.research_connection_score = analysis["research_connection_score"]
                        repo_analysis.implementation_maturity_score = analysis["implementation_maturity_score"]
                        repo_analysis.classification = analysis["classification"]
                        repo_analysis.evidence_json = json.dumps(analysis["evidence"], ensure_ascii=False)
                        repo_analysis.analyzed_at = datetime.now(timezone.utc).isoformat()
                        if readme or root_paths:
                            raw(session, "github_repository_contents", repo.full_name, {
                                "readme_excerpt": readme[:20000], "root_paths": root_paths,
                            })
                        existing = session.query(PaperRepo).filter_by(
                            paper_id=paper_id, repo_id=repo.id
                        ).first()
                        match_method = f"{evidence}|{repo_type}"
                        if existing:
                            existing.confidence = confidence
                            existing.match_method = match_method
                            existing.review_status = "needs_review"
                        else:
                            session.add(PaperRepo(
                                paper_id=paper_id, repo_id=repo.id,
                                match_method=match_method, confidence=confidence,
                                review_status="needs_review",
                            ))
                        raw(session, "github", repo_data["id"], repo_data)

                    session.commit()
                    if processed >= max_papers:
                        break

                cursor = (batch.get("meta") or {}).get("next_cursor")
                if not cursor:
                    break
        console.print(f"[green]Processed {processed} papers[/green]")
    finally:
        oa.close()
        cr.close()
        gh.close()
        pwc.close()


@cli.command()
def report(format: str = "table", output: str | None = None):
    """Display candidate paper-to-repository links."""
    Base.metadata.create_all(engine)
    with Session() as session:
        rows = session.query(
            Paper.title, Paper.doi, Paper.publication_date, Repo.full_name,
            Repo.stars, PaperRepo.confidence, PaperRepo.match_method,
            RepoAnalysis.research_connection_score,
            RepoAnalysis.implementation_maturity_score,
            RepoAnalysis.classification, RepoAnalysis.evidence_json,
            PaperRepo.review_status,
        ).join(PaperRepo, Paper.id == PaperRepo.paper_id).join(
            Repo, Repo.id == PaperRepo.repo_id
        ).outerjoin(
            RepoAnalysis,
            (Repo.id == RepoAnalysis.repo_id) & (Paper.id == RepoAnalysis.paper_id),
        ).order_by(
            RepoAnalysis.research_connection_score.desc().nullslast()
        ).all()

    headers = [
        "paper_title", "doi", "publication_date", "repository", "stars",
        "metadata_match_score", "metadata_evidence", "research_connection_score",
        "implementation_maturity_score", "repository_classification",
        "content_evidence", "review_status",
    ]
    if format == "csv":
        if not output:
            raise typer.BadParameter("--output is required with --format csv")
        with open(output, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)
        console.print(f"Exported {len(rows)} rows to {output}")
    else:
        table = Table(title="Paper-to-Repository Candidates")
        for header in headers:
            table.add_column(header, overflow="fold")
        for row in rows[:100]:
            values = list(row)
            if values[10]:
                try:
                    values[10] = "; ".join(json.loads(values[10]))
                except (TypeError, ValueError):
                    pass
            table.add_row(*(str(value if value is not None else "") for value in values))
        console.print(table)
        console.print(f"Total links: {len(rows)}")


if __name__ == "__main__":
    cli()
