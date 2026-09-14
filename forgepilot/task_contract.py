"""Deterministic single-file Current Task Contract primitives.

This module only validates and represents the supported Phase 1 Iteration 1
contract shape. Runtime activation and prompt projection are intentionally
handled by a later integration batch.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import PurePosixPath
from typing import Any, Mapping


SUPPORTED_CONTRACT_TOOLS = frozenset({"list_files", "read_file", "search", "write_file", "patch_file"})
OPAQUE_MUTATION_TOOLS = frozenset({"run_shell"})
SUPPORTED_MUTATION_TOOLS = frozenset({"write_file", "patch_file"})
REQUIRED_PROHIBITIONS = frozenset({"mutation_outside_target", "opaque_mutation"})


class ContractValidationError(ValueError):
    """Machine-readable validation failure for an explicit Contract input."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _as_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _as_string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of strings")
    result = tuple(_as_non_empty_string(item, field_name) for item in value)
    return result


def _normalize_target_path(value: Any) -> str:
    path = _as_non_empty_string(value, "target_path")
    if "\\" in path or "\x00" in path:
        raise ValueError("target_path must use a safe workspace-relative POSIX path")
    pure = PurePosixPath(path)
    if path == "." or pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ContractValidationError(
            "invalid_target_path",
            "target_path must be a normalized workspace-relative file path",
        )
    normalized = pure.as_posix()
    if normalized != path:
        raise ContractValidationError("invalid_target_path", "target_path must be normalized")
    return normalized


@dataclass(frozen=True)
class CompletionCondition:
    condition_id: str
    kind: str
    target_path: str
    expected_sha256: str
    expected_size: int


@dataclass(frozen=True)
class TaskContract:
    schema_version: int
    contract_id: str
    contract_version: str
    task_id: str
    goal: str
    workspace_scope: str
    target_path: str
    expected_content: str
    must_items: tuple[str, ...]
    completion_conditions: tuple[CompletionCondition, ...]
    allowed_tools: tuple[str, ...]
    prohibitions: tuple[str, ...]
    unresolved_conflicts: tuple[str, ...]
    user_pending_decisions: tuple[str, ...]
    provenance: tuple[tuple[str, str], ...]
    lifecycle_state: str = "active"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, task_id: str) -> "TaskContract":
        if not isinstance(data, Mapping):
            raise ValueError("task contract must be a mapping")
        if data.get("schema_version") != 1:
            raise ValueError("unsupported task contract schema_version")

        contract_id = _as_non_empty_string(data.get("contract_id"), "contract_id")
        contract_version = _as_non_empty_string(data.get("contract_version"), "contract_version")
        bound_task_id = _as_non_empty_string(task_id, "task_id")
        goal = _as_non_empty_string(data.get("goal"), "goal")
        if data.get("workspace_scope") != ".":
            raise ValueError("workspace_scope must be '.' for this iteration")
        target_path = _normalize_target_path(data.get("target_path"))

        condition_data = data.get("completion_condition")
        if not isinstance(condition_data, Mapping):
            raise ValueError("completion_condition must be a mapping")
        condition_id = _as_non_empty_string(condition_data.get("id"), "completion_condition.id")
        if condition_data.get("kind") != "file_exact_content":
            raise ValueError("only file_exact_content is supported")
        expected_content = condition_data.get("expected_content")
        if not isinstance(expected_content, str):
            raise ValueError("completion_condition.expected_content must be a string")
        expected_bytes = expected_content.encode("utf-8")
        condition = CompletionCondition(
            condition_id=condition_id,
            kind="file_exact_content",
            target_path=target_path,
            expected_sha256=sha256(expected_bytes).hexdigest(),
            expected_size=len(expected_bytes),
        )

        must_items = _as_string_tuple(data.get("must_items"), "must_items")
        if must_items != (condition_id,):
            raise ContractValidationError(
                "invalid_must_items",
                "must_items must contain exactly one hard completion-condition reference",
            )

        allowed_tools = _as_string_tuple(data.get("allowed_tools"), "allowed_tools")
        opaque_tools = set(allowed_tools) & OPAQUE_MUTATION_TOOLS
        if opaque_tools:
            raise ContractValidationError(
                "unsupported_opaque_tool",
                "opaque mutation tools are not supported in this Contract",
            )
        unsupported = set(allowed_tools) - SUPPORTED_CONTRACT_TOOLS
        if unsupported:
            raise ContractValidationError(
                "unsupported_tool",
                "allowed_tools contains an unsupported tool",
            )
        if len(set(allowed_tools)) != len(allowed_tools):
            raise ValueError("allowed_tools must not contain duplicates")

        prohibitions = _as_string_tuple(data.get("prohibitions"), "prohibitions")
        missing_prohibitions = REQUIRED_PROHIBITIONS - set(prohibitions)
        if missing_prohibitions:
            raise ContractValidationError(
                "missing_required_prohibition",
                "prohibitions must include target and opaque mutation safeguards",
            )
        unresolved_conflicts = _as_string_tuple(data.get("unresolved_conflicts"), "unresolved_conflicts")
        user_pending_decisions = _as_string_tuple(
            data.get("user_pending_decisions"), "user_pending_decisions"
        )
        provenance_data = data.get("provenance")
        if not isinstance(provenance_data, Mapping):
            raise ContractValidationError("missing_provenance", "provenance must be a mapping")
        if not {"source", "source_reference"}.issubset(provenance_data):
            raise ContractValidationError(
                "missing_provenance",
                "provenance must include source and source_reference",
            )
        provenance = tuple(
            (str(key), _as_non_empty_string(value, f"provenance.{key}"))
            for key, value in sorted(provenance_data.items())
        )
        if not provenance:
            raise ContractValidationError("missing_provenance", "provenance must not be empty")

        return cls(
            schema_version=1,
            contract_id=contract_id,
            contract_version=contract_version,
            task_id=bound_task_id,
            goal=goal,
            workspace_scope=".",
            target_path=target_path,
            expected_content=expected_content,
            must_items=must_items,
            completion_conditions=(condition,),
            allowed_tools=allowed_tools,
            prohibitions=prohibitions,
            unresolved_conflicts=unresolved_conflicts,
            user_pending_decisions=user_pending_decisions,
            provenance=provenance,
            lifecycle_state="active",
        )

    @property
    def is_execution_ready(self) -> bool:
        return not self.unresolved_conflicts and not self.user_pending_decisions

    @property
    def execution_not_ready(self) -> bool:
        return not self.is_execution_ready

    @property
    def expected_sha256(self) -> str:
        return self.completion_conditions[0].expected_sha256

    def summary(self) -> dict[str, Any]:
        """Return an artifact-safe summary without raw expected content."""
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "contract_version": self.contract_version,
            "task_id": self.task_id,
            "goal": self.goal,
            "workspace_scope": self.workspace_scope,
            "target_path": self.target_path,
            "must_items": list(self.must_items),
            "completion_conditions": [
                {
                    "id": condition.condition_id,
                    "kind": condition.kind,
                    "target_path": condition.target_path,
                    "expected_sha256": condition.expected_sha256,
                    "expected_size": condition.expected_size,
                }
                for condition in self.completion_conditions
            ],
            "allowed_tools": list(self.allowed_tools),
            "prohibitions": list(self.prohibitions),
            "unresolved_conflicts": list(self.unresolved_conflicts),
            "user_pending_decisions": list(self.user_pending_decisions),
            "provenance": dict(self.provenance),
            "execution_ready": self.is_execution_ready,
            "execution_not_ready": self.execution_not_ready,
            "lifecycle_state": self.lifecycle_state,
        }
