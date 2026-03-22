from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json
import random
from typing import Any


RUNOFF_LABEL = "მეორე ტური"
ABSENT_LABEL = "არ ესწრება"
PRESENT_LABEL = "ესწრება"


@dataclass(frozen=True)
class ModelData:
    candidate_options: list[str]
    electors: list[str]
    preference_headers: list[str]
    preferences: dict[str, dict[str, int]]
    default_parameters: dict[str, Any]


def load_model_data(json_path: str | Path | None = None) -> ModelData:
    """Load the election model extracted from the Excel workbook."""
    if json_path is None:
        json_path = Path(__file__).resolve().parent / "data" / "patriarch_model.json"

    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return ModelData(
        candidate_options=payload["candidate_options"],
        electors=payload["electors"],
        preference_headers=payload["preference_headers"],
        preferences=payload["preferences"],
        default_parameters=payload["default_parameters"],
    )


def _leftmost_max_choice(scores: list[int], selected_candidates: list[str]) -> str:
    """
    Reproduce Excel cell E2 logic exactly:
    IF(AND(B>=C,B>=D), B1, IF(C>=D, C1, D1))

    That means ties are broken left-to-right:
    candidate_1 > candidate_2 > candidate_3.
    """
    if scores[0] >= scores[1] and scores[0] >= scores[2]:
        return selected_candidates[0]
    if scores[1] >= scores[2]:
        return selected_candidates[1]
    return selected_candidates[2]


def simulate_single_election(
    model: ModelData,
    selected_candidates: list[str],
    voter_absence_probability: float,
    volatility_level: int,
    candidate_absence_probability: float,
    enable_voter_absence: bool = True,
    enable_candidate_absence: bool = True,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """
    Run one election using the same decision logic as the Excel model.

    Key fidelity rules preserved from Excel:
    - A selected candidate who is absent simply does not cast a vote themselves;
      their absence does NOT invalidate votes cast for them by others.
    - Score jitter is discrete and inclusive: RANDBETWEEN(-volatility, volatility)
    - Per-voter ties go to the leftmost selected candidate.
    - Final overall ties result in "მეორე ტური".
    """
    if rng is None:
        rng = random.Random()

    if len(selected_candidates) != 3:
        raise ValueError("Exactly three candidates must be selected.")
    if len(set(selected_candidates)) != 3:
        raise ValueError("Selected candidates must be distinct.")

    rows: list[dict[str, Any]] = []
    vote_counter: Counter[str] = Counter()

    for elector in model.electors:
        is_selected_candidate = elector in selected_candidates

        if is_selected_candidate:
            is_absent = enable_candidate_absence and rng.random() < candidate_absence_probability
        else:
            is_absent = enable_voter_absence and rng.random() < voter_absence_probability

        attendance = ABSENT_LABEL if is_absent else PRESENT_LABEL

        if is_absent:
            scores = [0, 0, 0]
            vote = ABSENT_LABEL
        else:
            scores = []
            for candidate in selected_candidates:
                base_score = int(model.preferences[elector][candidate])
                jitter = rng.randint(-volatility_level, volatility_level)
                scores.append(base_score + jitter)
            vote = _leftmost_max_choice(scores, selected_candidates)
            vote_counter[vote] += 1

        rows.append(
            {
                "მღვდელმთავარი": elector,
                "დასწრება": attendance,
                selected_candidates[0]: scores[0],
                selected_candidates[1]: scores[1],
                selected_candidates[2]: scores[2],
                "ხმა": vote,
            }
        )

    vote_totals = {candidate: vote_counter.get(candidate, 0) for candidate in selected_candidates}
    max_votes = max(vote_totals.values())
    ties_at_top = sum(1 for value in vote_totals.values() if value == max_votes)
    total_valid_votes = sum(vote_totals.values())

    if ties_at_top > 1:
        winner = RUNOFF_LABEL
    elif max_votes <= total_valid_votes / 2:
        winner = RUNOFF_LABEL
    else:
        winner = max(vote_totals, key=vote_totals.get)

    return {
        "selected_candidates": selected_candidates,
        "rows": rows,
        "vote_totals": vote_totals,
        "present_count": sum(1 for row in rows if row["დასწრება"] == PRESENT_LABEL),
        "absent_count": sum(1 for row in rows if row["დასწრება"] == ABSENT_LABEL),
        "winner": winner,
    }


def run_monte_carlo(
    model: ModelData,
    selected_candidates: list[str],
    iterations: int,
    voter_absence_probability: float,
    volatility_level: int,
    candidate_absence_probability: float,
    enable_voter_absence: bool = True,
    enable_candidate_absence: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("Iterations must be positive.")

    rng = random.Random(seed)
    winners: list[str] = []
    aggregate_vote_totals = {candidate: [] for candidate in selected_candidates}
    present_counts: list[int] = []
    absent_counts: list[int] = []

    for _ in range(iterations):
        result = simulate_single_election(
            model=model,
            selected_candidates=selected_candidates,
            voter_absence_probability=voter_absence_probability,
            volatility_level=volatility_level,
            candidate_absence_probability=candidate_absence_probability,
            enable_voter_absence=enable_voter_absence,
            enable_candidate_absence=enable_candidate_absence,
            rng=rng,
        )
        winners.append(result["winner"])
        present_counts.append(result["present_count"])
        absent_counts.append(result["absent_count"])
        for candidate, votes in result["vote_totals"].items():
            aggregate_vote_totals[candidate].append(votes)

    winner_counts = Counter(winners)
    winner_probabilities = {
        candidate: winner_counts.get(candidate, 0) / iterations
        for candidate in [*selected_candidates, RUNOFF_LABEL]
    }

    vote_statistics = {
        candidate: {
            "average_votes": sum(votes) / len(votes),
            "min_votes": min(votes),
            "max_votes": max(votes),
        }
        for candidate, votes in aggregate_vote_totals.items()
    }

    return {
        "iterations": iterations,
        "selected_candidates": selected_candidates,
        "winner_counts": dict(winner_counts),
        "winner_probabilities": winner_probabilities,
        "vote_statistics": vote_statistics,
        "average_present_count": sum(present_counts) / len(present_counts),
        "average_absent_count": sum(absent_counts) / len(absent_counts),
        "winners": winners,
    }
