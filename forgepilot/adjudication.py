"""Pure deterministic admission primitives for the supported task class."""

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping

from .task_contract import OPAQUE_MUTATION_TOOLS, SUPPORTED_MUTATION_TOOLS, TaskContract


@dataclass(frozen=True)
class ActionCandidate:
    candidate_id: str
    tool_name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class AdmissionDecision:
    candidate_id: str
    admitted: bool
    reason_code: str
    constraint_id: str = ""
    normalized_path: str = ""


def _candidate_path(candidate: ActionCandidate) -> str | None:
    value = candidate.arguments.get("path")
    if not isinstance(value, str) or not value:
        return None
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        return None
    return pure.as_posix()


def admit_action(contract: TaskContract, candidate: ActionCandidate) -> AdmissionDecision:
    """Return a deterministic allow/block decision without executing anything."""

    if not contract.is_execution_ready:
        return AdmissionDecision(
            candidate_id=candidate.candidate_id,
            admitted=False,
            reason_code="contract_not_execution_ready",
            constraint_id="execution_ready",
        )

    if candidate.tool_name in OPAQUE_MUTATION_TOOLS:
        return AdmissionDecision(
            candidate_id=candidate.candidate_id,
            admitted=False,
            reason_code="opaque_mutation_scope",
            constraint_id="opaque_mutation",
        )

    if candidate.tool_name not in contract.allowed_tools:
        return AdmissionDecision(
            candidate_id=candidate.candidate_id,
            admitted=False,
            reason_code="tool_not_allowed",
            constraint_id="allowed_tools",
        )

    if candidate.tool_name in SUPPORTED_MUTATION_TOOLS:
        normalized_path = _candidate_path(candidate)
        if normalized_path is None:
            return AdmissionDecision(
                candidate_id=candidate.candidate_id,
                admitted=False,
                reason_code="mutation_path_not_determinable",
                constraint_id="target_path",
            )
        if normalized_path != contract.target_path:
            return AdmissionDecision(
                candidate_id=candidate.candidate_id,
                admitted=False,
                reason_code="off_target_mutation",
                constraint_id="target_path",
                normalized_path=normalized_path,
            )
        return AdmissionDecision(
            candidate_id=candidate.candidate_id,
            admitted=True,
            reason_code="admitted",
            normalized_path=normalized_path,
        )

    return AdmissionDecision(
        candidate_id=candidate.candidate_id,
        admitted=True,
        reason_code="admitted",
        normalized_path=_candidate_path(candidate) or "",
    )


@dataclass(frozen=True)
class CompletionCandidate:
    candidate_id: str
    text: str


@dataclass(frozen=True)
class CompletionVerdict:
    result: str
    candidate_id: str
    covered_condition_ids: tuple[str, ...]
    unmet_condition_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    reason_code: str


def evaluate_completion(contract: TaskContract, ledger: Any, candidate: CompletionCandidate) -> CompletionVerdict:
    """Adjudicate a final candidate from current eligible verified Evidence only."""

    condition_ids = tuple(condition.condition_id for condition in contract.completion_conditions)
    if not contract.is_execution_ready:
        return CompletionVerdict(
            result="contract_blocked",
            candidate_id=candidate.candidate_id,
            covered_condition_ids=(),
            unmet_condition_ids=condition_ids,
            evidence_ids=(),
            reason_code="contract_not_execution_ready",
        )

    evidence = ledger.current_evidence()
    if evidence is None:
        return CompletionVerdict(
            result="evidence_insufficient",
            candidate_id=candidate.candidate_id,
            covered_condition_ids=(),
            unmet_condition_ids=condition_ids,
            evidence_ids=(),
            reason_code="current_evidence_insufficient",
        )

    return CompletionVerdict(
        result="verified_completed",
        candidate_id=candidate.candidate_id,
        covered_condition_ids=(evidence.condition_id,),
        unmet_condition_ids=(),
        evidence_ids=(evidence.evidence_id,),
        reason_code="current_evidence_verified",
    )
