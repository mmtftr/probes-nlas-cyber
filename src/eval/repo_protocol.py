"""Repo-level evaluation for scan leads.

The scanner emits ranked line-window leads. Repo benchmark manifests contain
ground-truth vulnerable line ranges. This module scores whether the top-K scan
budget actually surfaces those vulnerable regions.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class RepoMetrics:
    repo_id: str
    n_traces: int
    n_leads: int
    first_hit_rank: Optional[int]
    reciprocal_rank: float
    repo_recall_at_k: dict[str, float] = field(default_factory=dict)
    trace_recall_at_k: dict[str, float] = field(default_factory=dict)
    precision_at_k: dict[str, float] = field(default_factory=dict)
    cwe_top1_on_hits_at_k: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RepoEvalReport:
    n_repos: int
    k_values: list[int]
    mean_reciprocal_rank: float
    repo_recall_at_k: dict[str, float]
    trace_recall_at_k: dict[str, float]
    precision_at_k: dict[str, float]
    cwe_top1_on_hits_at_k: dict[str, float]
    repos: list[RepoMetrics]

    def to_dict(self) -> dict:
        return {
            "n_repos": self.n_repos,
            "k_values": self.k_values,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "repo_recall_at_k": self.repo_recall_at_k,
            "trace_recall_at_k": self.trace_recall_at_k,
            "precision_at_k": self.precision_at_k,
            "cwe_top1_on_hits_at_k": self.cwe_top1_on_hits_at_k,
            "repos": [r.to_dict() for r in self.repos],
        }


def load_manifest_index(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_leads_jsonl(path: Path) -> list[dict]:
    leads: list[dict] = []
    if not path.exists():
        return leads
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                leads.append(json.loads(line))
    return sorted(leads, key=lambda lead: int(lead.get("rank", 10**9)))


def line_ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and a_end >= b_start


def lead_hits_trace(lead: dict, trace: dict) -> bool:
    if lead.get("file") != trace.get("file"):
        return False
    try:
        return line_ranges_overlap(
            int(lead["start_line"]),
            int(lead["end_line"]),
            int(trace["start_line"]),
            int(trace["end_line"]),
        )
    except (KeyError, TypeError, ValueError):
        return False


def hit_trace_indices(leads: Iterable[dict], traces: list[dict]) -> set[int]:
    hits: set[int] = set()
    for lead in leads:
        for i, trace in enumerate(traces):
            if lead_hits_trace(lead, trace):
                hits.add(i)
    return hits


def lead_is_hit(lead: dict, traces: list[dict]) -> bool:
    return any(lead_hits_trace(lead, trace) for trace in traces)


def cwe_match_on_hits(leads: Iterable[dict], traces: list[dict]) -> Optional[float]:
    total = 0
    matched = 0
    for lead in leads:
        lead_cwe = lead.get("probe_top_cwe")
        if not lead_cwe:
            continue
        for trace in traces:
            if not lead_hits_trace(lead, trace):
                continue
            trace_cwe = trace.get("cwe")
            if not trace_cwe:
                continue
            total += 1
            matched += int(str(lead_cwe) == str(trace_cwe))
            break
    if total == 0:
        return None
    return matched / total


def evaluate_repo(manifest: dict, leads: list[dict], *, k_values: Iterable[int] = (1, 5, 10, 25)) -> RepoMetrics:
    traces = list(manifest.get("vulnerability_traces") or [])
    first_hit_rank: Optional[int] = None
    for i, lead in enumerate(leads, start=1):
        if lead_is_hit(lead, traces):
            first_hit_rank = i
            break

    metrics = RepoMetrics(
        repo_id=str(manifest.get("repo_id")),
        n_traces=len(traces),
        n_leads=len(leads),
        first_hit_rank=first_hit_rank,
        reciprocal_rank=(1.0 / first_hit_rank) if first_hit_rank else 0.0,
    )
    for k in k_values:
        key = f"@{int(k)}"
        top = leads[: int(k)]
        trace_hits = hit_trace_indices(top, traces)
        lead_hits = sum(1 for lead in top if lead_is_hit(lead, traces))
        metrics.repo_recall_at_k[key] = 1.0 if trace_hits else 0.0
        metrics.trace_recall_at_k[key] = (len(trace_hits) / len(traces)) if traces else 0.0
        metrics.precision_at_k[key] = (lead_hits / len(top)) if top else 0.0
        cwe_acc = cwe_match_on_hits(top, traces)
        metrics.cwe_top1_on_hits_at_k[key] = cwe_acc if cwe_acc is not None else float("nan")
    return metrics


def full_repo_report(
    manifests: list[dict],
    leads_by_repo: dict[str, list[dict]],
    *,
    k_values: Iterable[int] = (1, 5, 10, 25),
) -> RepoEvalReport:
    ks = [int(k) for k in k_values]
    repos = [
        evaluate_repo(m, leads_by_repo.get(str(m.get("repo_id")), []), k_values=ks)
        for m in manifests
    ]

    def mean_metric(attr: str, key: str) -> float:
        if not repos:
            return 0.0
        vals = [getattr(repo, attr)[key] for repo in repos]
        vals = [v for v in vals if v == v]
        return sum(vals) / len(vals) if vals else float("nan")

    keys = [f"@{k}" for k in ks]
    return RepoEvalReport(
        n_repos=len(repos),
        k_values=ks,
        mean_reciprocal_rank=sum(r.reciprocal_rank for r in repos) / len(repos) if repos else 0.0,
        repo_recall_at_k={key: mean_metric("repo_recall_at_k", key) for key in keys},
        trace_recall_at_k={key: mean_metric("trace_recall_at_k", key) for key in keys},
        precision_at_k={key: mean_metric("precision_at_k", key) for key in keys},
        cwe_top1_on_hits_at_k={key: mean_metric("cwe_top1_on_hits_at_k", key) for key in keys},
        repos=repos,
    )
