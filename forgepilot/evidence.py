"""Minimal observation-to-Evidence model for deterministic single-file tasks."""

from dataclasses import dataclass
from typing import Optional

from .task_contract import TaskContract


@dataclass(frozen=True)
class ToolObservation:
    observation_id: str
    task_id: str
    contract_version: str
    candidate_id: str
    tool_name: str
    status: str
    execution_attempted: bool
    target_path: str
    target_exists: Optional[bool]
    target_sha256: str
    target_size: int
    target_revision: int = 0
    target_before_exists: Optional[bool] = None
    target_before_sha256: str = ""
    target_before_size: int = 0


@dataclass
class EvidenceRecord:
    evidence_id: str
    observation_id: str
    condition_id: str
    eligible: bool
    verified: bool
    match: str
    freshness: str = "current"
    target_revision: int = 0
    reason_code: str = ""
    superseded_by: str = ""
    resolves_observation_ids: tuple[str, ...] = ()


class EvidenceLedger:
    def __init__(self, contract: TaskContract):
        self.contract = contract
        self._records: list[EvidenceRecord] = []
        self._unresolved_observation_ids: list[str] = []
        self._last_target_state: tuple[Optional[bool], str, int] | None = None
        self._target_revision = 0

    @property
    def records(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._records)

    @property
    def target_revision(self) -> int:
        return self._target_revision

    def _state_for(self, observation: ToolObservation) -> tuple[Optional[bool], str, int]:
        return (observation.target_exists, observation.target_sha256, observation.target_size)

    def _advance_target_state(self, observation: ToolObservation) -> int:
        state = self._state_for(observation)
        if self._last_target_state is not None and state != self._last_target_state:
            self._target_revision += 1
            for record in self._records:
                if record.freshness == "current" and record.verified:
                    record.freshness = "stale"
                    record.superseded_by = observation.observation_id
        self._last_target_state = state
        return self._target_revision

    def record(self, observation: ToolObservation) -> EvidenceRecord:
        condition = self.contract.completion_conditions[0]

        common = {
            "observation_id": observation.observation_id,
            "condition_id": condition.condition_id,
        }
        if observation.task_id != self.contract.task_id:
            record = EvidenceRecord(
                evidence_id=f"evidence-{len(self._records) + 1:03d}",
                eligible=False,
                verified=False,
                match="unknown",
                reason_code="task_mismatch",
                target_revision=self._target_revision,
                **common,
            )
            self._records.append(record)
            return record
        if observation.contract_version != self.contract.contract_version:
            record = EvidenceRecord(
                evidence_id=f"evidence-{len(self._records) + 1:03d}",
                eligible=False,
                verified=False,
                match="unknown",
                reason_code="contract_version_mismatch",
                target_revision=self._target_revision,
                **common,
            )
            self._records.append(record)
            return record
        if observation.target_path != condition.target_path:
            record = EvidenceRecord(
                evidence_id=f"evidence-{len(self._records) + 1:03d}",
                eligible=False,
                verified=False,
                match="unknown",
                reason_code="target_mismatch",
                target_revision=self._target_revision,
                **common,
            )
            self._records.append(record)
            return record

        # Only a validated observation for this task, contract version, and
        # condition target may advance target revision or stale current evidence.
        self._advance_target_state(observation)

        is_success = observation.status == "success" and observation.execution_attempted
        if not is_success:
            if observation.execution_attempted and observation.status in {"error", "partial"}:
                self._unresolved_observation_ids.append(observation.observation_id)
            record = EvidenceRecord(
                evidence_id=f"evidence-{len(self._records) + 1:03d}",
                eligible=False,
                verified=False,
                match="unknown",
                reason_code=f"observation_{observation.status}",
                target_revision=self._target_revision,
                **common,
            )
            self._records.append(record)
            return record

        exact = (
            observation.target_exists is True
            and observation.target_sha256 == condition.expected_sha256
            and observation.target_size == condition.expected_size
        )
        resolved = tuple(self._unresolved_observation_ids)
        self._unresolved_observation_ids.clear()
        record = EvidenceRecord(
            evidence_id=f"evidence-{len(self._records) + 1:03d}",
            eligible=True,
            verified=True,
            match="exact" if exact else "mismatch",
            reason_code="exact_content" if exact else "content_mismatch",
            target_revision=self._target_revision,
            resolves_observation_ids=resolved,
            **common,
        )
        self._records.append(record)
        return record

    def current_evidence(self) -> EvidenceRecord | None:
        if self._unresolved_observation_ids:
            return None
        for record in reversed(self._records):
            if (
                record.freshness == "current"
                and record.eligible
                and record.verified
                and record.match == "exact"
            ):
                return record
        return None
