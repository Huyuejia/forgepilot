"""Agent 运行时核心逻辑。

ForgePilot 就是包在模型外面的控制循环：负责组 prompt、解析模型输出、
校验并执行工具、写 trace、更新工作记忆，以及在合适的时候停下来。
"""

import json
import os
import re
import textwrap
import uuid
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import memory as memorylib
from . import planner
from .adjudication import ActionCandidate, CompletionCandidate, admit_action, evaluate_completion
from .context_manager import ContextManager, ProtectedContractBudgetError
from .evidence import EvidenceLedger, ToolObservation
from .run_store import RunStore
from .task_state import TaskState
from .task_contract import ContractValidationError, TaskContract
from .models import ModelExhaustedError
from . import tools as toolkit
from .workspace import IGNORED_PATH_NAMES, MAX_HISTORY, WorkspaceContext, clip, now

SENSITIVE_ENV_NAME_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
REDACTED_VALUE = "<redacted>"
DEFAULT_SHELL_ENV_ALLOWLIST = ("HOME", "LANG", "LC_ALL", "LC_CTYPE", "LOGNAME", "PATH", "PWD", "SHELL", "TERM", "TMPDIR", "TMP", "TEMP", "USER")
DEFAULT_FEATURE_FLAGS = {
    "memory": True,
    "relevant_memory": True,
    "context_reduction": True,
    "prompt_cache": True,
}
CHECKPOINT_SCHEMA_VERSION = "phase1-v1"
CHECKPOINT_NONE_STATUS = "no-checkpoint"
CHECKPOINT_FULL_VALID_STATUS = "full-valid"
CHECKPOINT_PARTIAL_STALE_STATUS = "partial-stale"
CHECKPOINT_WORKSPACE_MISMATCH_STATUS = "workspace-mismatch"
CHECKPOINT_SCHEMA_MISMATCH_STATUS = "schema-mismatch"
COMPATIBILITY_MODE = "compatibility"
ENHANCED_CONTRACT_MODE = "enhanced_contract"
DURABLE_MEMORY_INTENT_PATTERN = re.compile(r"(?i)\b(capture|remember|save|store|persist|note)\b")
DURABLE_MEMORY_INTENT_ZH_PATTERN = re.compile(r"(记住|保存|记录|沉淀|长期记忆|持久记忆)")
DURABLE_MEMORY_LINE_PATTERNS = (
    ("project-conventions", re.compile(r"(?i)^Project convention:\s*(.+)$")),
    ("key-decisions", re.compile(r"(?i)^Decision:\s*(.+)$")),
    ("dependency-facts", re.compile(r"(?i)^Dependency:\s*(.+)$")),
    ("user-preferences", re.compile(r"(?i)^Preference:\s*(.+)$")),
    ("project-conventions", re.compile(r"^项目约定：\s*(.+)$")),
    ("key-decisions", re.compile(r"^决策：\s*(.+)$")),
    ("dependency-facts", re.compile(r"^依赖：\s*(.+)$")),
    ("user-preferences", re.compile(r"^偏好：\s*(.+)$")),
)
SECRET_SHAPED_TEXT_PATTERN = re.compile(r"(?i)(\b(api[_ -]?key|token|secret|password)\b|sk-[A-Za-z0-9_-]{6,})")


def _legacy_contract_validation_code(error):
    """Map known legacy validator ValueErrors to stable activation codes."""
    message = str(error).lower()
    if "schema_version" in message:
        return "unsupported_schema_version"
    if "target_path" in message:
        return "invalid_target_path"
    if "completion_condition" in message:
        return "invalid_completion_condition"
    if "must_items" in message:
        return "invalid_must_items"
    if "allowed_tools" in message:
        return "unsupported_tool"
    if "provenance" in message:
        return "missing_provenance"
    if "workspace_scope" in message:
        return "invalid_workspace_scope"
    if "task contract" in message:
        return "invalid_contract_shape"
    return "legacy_contract_validation_error"


@dataclass
class PromptPrefix:
    # prefix 除了文本本身，还带一小份元数据，
    # 这样 runtime 才能明确判断 prefix 是否可以复用。
    text: str
    hash: str
    workspace_fingerprint: str
    tool_signature: str
    built_at: str


class SessionStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, session_id):
        return self.root / f"{session_id}.json"

    def save(self, session):
        path = self.path(session["id"])
        path.write_text(json.dumps(session, indent=2), encoding="utf-8")
        return path

    def load(self, session_id):
        return json.loads(self.path(session_id).read_text(encoding="utf-8"))

    def latest(self):
        files = sorted(self.root.glob("*.json"), key=lambda path: path.stat().st_mtime)
        return files[-1].stem if files else None


class ForgePilot:
    def __init__(
        self,
        model_client,
        workspace,
        session_store,
        session=None,
        run_store=None,
        approval_policy="ask",
        max_steps=6,
        max_new_tokens=512,
        depth=0,
        max_depth=1,
        read_only=False,
        shell_env_allowlist=None,
        secret_env_names=None,
        feature_flags=None,
    ):
        self.model_client = model_client
        self.workspace = workspace
        self.root = Path(workspace.repo_root)
        self.session_store = session_store
        self.approval_policy = approval_policy
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.depth = depth
        self.max_depth = max_depth
        self.read_only = read_only
        self.shell_env_allowlist = tuple(shell_env_allowlist or DEFAULT_SHELL_ENV_ALLOWLIST)
        self.secret_env_names = {str(name).upper() for name in (secret_env_names or ())}
        self.feature_flags = dict(DEFAULT_FEATURE_FLAGS)
        if feature_flags:
            self.feature_flags.update({str(key): bool(value) for key, value in feature_flags.items()})
        self.run_store = run_store or RunStore(Path(workspace.repo_root) / ".ForgePilot" / "runs")
        self.session = session or {
            "id": datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
            "created_at": now(),
            "workspace_root": workspace.repo_root,
            "history": [],
            "memory": memorylib.default_memory_state(),
        }
        self._ensure_session_shape()
        self.memory = memorylib.LayeredMemory(
            self.session.setdefault("memory", memorylib.default_memory_state()),
            workspace_root=self.root,
        )
        self.session["memory"] = self.memory.to_dict()
        self.tools = self.build_tools()
        self.prefix_state = self.build_prefix()
        self.prefix = self.prefix_state.text
        self.context_manager = ContextManager(self)
        self.resume_state = self.evaluate_resume_state()
        self.session_path = self.session_store.save(self.session)
        self.current_task_state = None
        self.current_run_dir = None
        self.last_prompt_metadata = {}
        self.last_completion_metadata = {}
        self.last_durable_promotions = []
        self.last_durable_rejections = []
        self.last_durable_superseded = []
        self._last_tool_result_metadata = {}
        self.execution_mode = None
        self.active_contract = None
        self.contract_activation = {}
        self.last_activation_failure = {}
        self._action_candidate_sequence = 0
        self._completion_candidate_sequence = 0
        self._observation_sequence = 0
        self.evidence_ledger = None
        self._last_prefix_refresh = {
            "workspace_changed": False,
            "prefix_changed": False,
        }

    @classmethod
    def from_session(cls, model_client, workspace, session_store, session_id, **kwargs):
        return cls(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session=session_store.load(session_id),
            **kwargs,
        )

    def _ensure_session_shape(self):
        self.session.setdefault("history", [])
        self.session.setdefault("memory", memorylib.default_memory_state())
        checkpoints = self.session.setdefault("checkpoints", {})
        if not isinstance(checkpoints, dict):
            checkpoints = {}
            self.session["checkpoints"] = checkpoints
        checkpoints.setdefault("current_id", "")
        checkpoints.setdefault("items", {})
        runtime_identity = self.session.setdefault("runtime_identity", {})
        if not isinstance(runtime_identity, dict):
            self.session["runtime_identity"] = {}
        resume_state = self.session.setdefault("resume_state", {})
        if not isinstance(resume_state, dict):
            self.session["resume_state"] = {}
        self.session["plan"] = planner.normalize_plan_state(self.session.get("plan"))

    def current_runtime_identity(self):
        return {
            "session_id": self.session.get("id", ""),
            "cwd": str(self.root),
            "model": str(getattr(self.model_client, "model", "")),
            "model_client": self.model_client.__class__.__name__,
            "approval_policy": self.approval_policy,
            "read_only": bool(self.read_only),
            "max_steps": int(self.max_steps),
            "max_new_tokens": int(self.max_new_tokens),
            "feature_flags": dict(self.feature_flags),
            "shell_env_allowlist": list(self.shell_env_allowlist),
            "workspace_fingerprint": getattr(getattr(self, "prefix_state", None), "workspace_fingerprint", self.workspace.fingerprint()),
            "tool_signature": self.tool_signature(),
        }

    def checkpoint_state(self):
        self._ensure_session_shape()
        return self.session["checkpoints"]

    def current_checkpoint(self):
        state = self.checkpoint_state()
        checkpoint_id = str(state.get("current_id", "")).strip()
        if not checkpoint_id:
            return None
        return state.get("items", {}).get(checkpoint_id)

    def invalidate_stale_memory(self):
        invalidated = self.memory.invalidate_stale_file_summaries()
        self.session["memory"] = self.memory.to_dict()
        return invalidated

    def evaluate_resume_state(self):
        previous_resume_state = dict(self.session.get("resume_state", {}) or {})
        invalidated = self.invalidate_stale_memory()
        checkpoint = self.current_checkpoint()
        status = CHECKPOINT_NONE_STATUS
        stale_paths = list(invalidated)
        mismatch_fields = []
        if checkpoint:
            if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                status = CHECKPOINT_SCHEMA_MISMATCH_STATUS
            else:
                for item in checkpoint.get("key_files", []):
                    path = str(item.get("path", "")).strip()
                    if not path:
                        continue
                    expected = item.get("freshness")
                    current = memorylib.file_freshness(path, self.root)
                    if expected != current and path not in stale_paths:
                        stale_paths.append(path)
                saved_identity = dict(checkpoint.get("runtime_identity", {}) or self.session.get("runtime_identity", {}) or {})
                current_identity = self.current_runtime_identity()
                identity_keys = (
                    "cwd",
                    "model",
                    "model_client",
                    "approval_policy",
                    "read_only",
                    "max_steps",
                    "max_new_tokens",
                    "feature_flags",
                    "shell_env_allowlist",
                    "workspace_fingerprint",
                    "tool_signature",
                )
                for key in identity_keys:
                    if key not in saved_identity:
                        continue
                    if saved_identity.get(key) != current_identity.get(key):
                        mismatch_fields.append(key)
                mismatch_fields.sort()
                if stale_paths:
                    status = CHECKPOINT_PARTIAL_STALE_STATUS
                elif mismatch_fields:
                    status = CHECKPOINT_WORKSPACE_MISMATCH_STATUS
                else:
                    status = CHECKPOINT_FULL_VALID_STATUS

        resume_state = {
            "status": status,
            "stale_paths": stale_paths,
            "runtime_identity_mismatch_fields": mismatch_fields,
            "stale_summary_invalidations": max(
                len(invalidated),
                int(previous_resume_state.get("stale_summary_invalidations", 0))
                if status == CHECKPOINT_PARTIAL_STALE_STATUS
                else 0,
            ),
        }
        self.session["resume_state"] = resume_state
        self.session["runtime_identity"] = self.current_runtime_identity()
        return resume_state

    def render_checkpoint_text(self):
        checkpoint = self.current_checkpoint()
        if not checkpoint:
            return ""
        lines = [
            "Task checkpoint:",
            f"- Resume status: {self.resume_state.get('status', CHECKPOINT_NONE_STATUS)}",
            f"- Current goal: {checkpoint.get('current_goal', '-') or '-'}",
            f"- Current blocker: {checkpoint.get('current_blocker', '-') or '-'}",
            f"- Next step: {checkpoint.get('next_step', '-') or '-'}",
        ]
        key_files = [str(item.get("path", "")).strip() for item in checkpoint.get("key_files", []) if str(item.get("path", "")).strip()]
        lines.append(f"- Key files: {', '.join(key_files) or '-'}")
        if checkpoint.get("completed"):
            lines.append("- Completed: " + " | ".join(str(item) for item in checkpoint.get("completed", [])))
        if checkpoint.get("excluded"):
            lines.append("- Excluded: " + " | ".join(str(item) for item in checkpoint.get("excluded", [])))
        if self.resume_state.get("stale_paths"):
            lines.append("- Stale paths: " + ", ".join(self.resume_state["stale_paths"]))
        summary = str(checkpoint.get("summary", "")).strip()
        if summary:
            lines.append(f"- Summary: {summary}")
        return "\n".join(lines)

    @staticmethod
    def remember(bucket, item, limit):
        if not item:
            return
        if item in bucket:
            bucket.remove(item)
        bucket.append(item)
        del bucket[:-limit]

    def build_tools(self):
        return toolkit.build_tool_registry(self)

    def tool_signature(self):
        payload = []
        for name in sorted(self.tools):
            tool = self.tools[name]
            payload.append(
                {
                    "name": name,
                    "schema": tool["schema"],
                    "risky": tool["risky"],
                    "description": tool["description"],
                }
            )
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def build_prefix(self):
        tool_lines = []
        for name, tool in self.tools.items():
            fields = ", ".join(f"{key}: {value}" for key, value in tool["schema"].items())
            risk = "approval required" if tool["risky"] else "safe"
            tool_lines.append(f"- {name}({fields}) [{risk}] {tool['description']}")
        tool_text = "\n".join(tool_lines)
        examples = "\n".join(
            [
                '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
                '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
                '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
                '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
                '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
                '<tool>{"name":"update_plan","args":{"items":[{"text":"Inspect code","status":"completed"},{"text":"Run tests","status":"in_progress"}]}}</tool>',
                "<final>Done.</final>",
            ]
        )
        # prefix 可以理解成 agent 的“工作手册”：
        # 它是谁、工具怎么调用、当前仓库是什么状态，都写在这里。
        text = textwrap.dedent(
            f"""\
            You are ForgePilot, a small local coding agent working inside a local repository.

            Rules:
            - Use tools instead of guessing about the workspace.
            - Return exactly one <tool>...</tool> or one <final>...</final>.
            - Tool calls must look like:
              <tool>{{"name":"tool_name","args":{{...}}}}</tool>
            - For write_file and patch_file with multi-line text, prefer XML style:
              <tool name="write_file" path="file.py"><content>...</content></tool>
            - Final answers must look like:
              <final>your answer</final>
            - Never invent tool results.
            - Keep answers concise and concrete.
            - If the user asks you to create or update a specific file and the path is clear, use write_file or patch_file instead of repeatedly listing files.
            - Before writing tests for existing code, read the implementation first.
            - When writing tests, match the current implementation unless the user explicitly asked you to change the code.
            - New files should be complete and runnable, including obvious imports.
            - Do not repeat the same tool call with the same arguments if it did not help. Choose a different tool or return a final answer.
            - Required tool arguments must not be empty. Do not call read_file, write_file, patch_file, run_shell, or delegate with args={{}}.
            - Use update_plan when a task has multiple steps, when progress changes, or when you become blocked.
            - Keep exactly one plan item in_progress unless all items are completed or blocked.

            Tools:
            {tool_text}

            Valid response examples:
            {examples}

            {self.workspace.text()}
            """
        ).strip()
        return PromptPrefix(
            text=text,
            hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            workspace_fingerprint=self.workspace.fingerprint(),
            tool_signature=self.tool_signature(),
            built_at=now(),
        )

    def _apply_prefix_state(self, prefix_state):
        self.prefix_state = prefix_state
        self.prefix = prefix_state.text

    def refresh_prefix(self, force=False):
        previous_hash = getattr(getattr(self, "prefix_state", None), "hash", None)
        previous_workspace_fingerprint = getattr(getattr(self, "prefix_state", None), "workspace_fingerprint", None)

        # 工作区事实相对稳定，所以这里按整体刷新；
        # 只有这些事实真的变化了，才重建完整 prefix。
        refreshed_workspace = WorkspaceContext.build(self.root)
        refreshed_workspace_fingerprint = refreshed_workspace.fingerprint()
        workspace_changed = force or refreshed_workspace_fingerprint != previous_workspace_fingerprint
        if workspace_changed:
            self.workspace = refreshed_workspace

        prefix_state = self.build_prefix() if workspace_changed or force or previous_hash is None else self.prefix_state
        prefix_changed = force or previous_hash != prefix_state.hash
        if prefix_changed:
            self._apply_prefix_state(prefix_state)

        self._last_prefix_refresh = {
            "workspace_changed": workspace_changed,
            "prefix_changed": prefix_changed,
        }
        return dict(self._last_prefix_refresh)

    def memory_text(self):
        return self.memory.render_memory_text()

    def plan_text(self):
        return planner.render_plan_text(self.session.get("plan"))

    def skill_text(self, user_message):
        from . import skills

        return skills.render_relevant_skills(user_message)

    def set_plan_items(self, items):
        self.session["plan"] = planner.set_plan_items(self.session.get("plan"), items)
        self.session_path = self.session_store.save(self.session)
        return self.session["plan"]

    def update_plan_item_status(self, item_id, status):
        self.session["plan"] = planner.update_plan_item_status(self.session.get("plan"), item_id, status)
        self.session_path = self.session_store.save(self.session)
        return self.session["plan"]

    def history_text(self):
        history = self.session["history"]
        if not history:
            return "- empty"

        lines = []
        seen_reads = set()
        recent_start = max(0, len(history) - 6)
        for index, item in enumerate(history):
            recent = index >= recent_start
            if item["role"] == "tool" and item["name"] == "read_file" and not recent:
                path = str(item["args"].get("path", ""))
                if path in seen_reads:
                    continue
                seen_reads.add(path)

            if item["role"] == "tool":
                limit = 900 if recent else 180
                lines.append(f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}")
                lines.append(clip(item["content"], limit))
            else:
                limit = 900 if recent else 220
                lines.append(f"[{item['role']}] {clip(item['content'], limit)}")

        return clip("\n".join(lines), MAX_HISTORY)

    def feature_enabled(self, name):
        return bool(self.feature_flags.get(str(name), False))

    def prompt(self, user_message):
        prompt, _ = self._build_prompt_and_metadata(user_message)
        return prompt

    def record(self, item):
        self.session["history"].append(item)
        self.session_path = self.session_store.save(self.session)

    @staticmethod
    def looks_sensitive_env_name(name):
        upper = str(name).upper()
        return any(upper == marker or upper.endswith(marker) or upper.endswith(f"_{marker}") for marker in SENSITIVE_ENV_NAME_MARKERS)

    def is_secret_env_name(self, name):
        upper = str(name).upper()
        return upper in self.secret_env_names or self.looks_sensitive_env_name(upper)

    def configured_secret_env_items(self):
        items = [
            (name, value)
            for name, value in os.environ.items()
            if str(name).upper() in self.secret_env_names and value
        ]
        items.sort(key=lambda item: item[0])
        return items

    def detected_secret_env_items(self):
        items = [
            (name, value)
            for name, value in os.environ.items()
            if self.is_secret_env_name(name) and value
        ]
        items.sort(key=lambda item: item[0])
        return items

    def secret_env_summary(self):
        names = [name for name, _ in self.configured_secret_env_items()]
        return {
            "secret_env_count": len(names),
            "secret_env_names": names,
        }

    def detected_secret_env_summary(self):
        names = [name for name, _ in self.detected_secret_env_items()]
        return {
            "secret_env_count": len(names),
            "secret_env_names": names,
        }

    def redact_text(self, text):
        text = str(text)
        for _, value in sorted(self.detected_secret_env_items(), key=lambda item: len(item[1]), reverse=True):
            text = text.replace(value, REDACTED_VALUE)
        return text

    def redact_artifact(self, value, key=None):
        if key and self.is_secret_env_name(key):
            return REDACTED_VALUE
        if isinstance(value, dict):
            return {
                str(item_key): self.redact_artifact(item_value, key=item_key)
                for item_key, item_value in value.items()
            }
        if isinstance(value, list):
            return [self.redact_artifact(item, key=key) for item in value]
        if isinstance(value, tuple):
            return [self.redact_artifact(item, key=key) for item in value]
        if isinstance(value, str):
            redacted = self.redact_text(value)
            return redacted
        return value

    def shell_env(self):
        env = {
            name: os.environ[name]
            for name in self.shell_env_allowlist
            if name in os.environ
        }
        env["PWD"] = str(self.root)
        if "PATH" not in env and os.environ.get("PATH"):
            env["PATH"] = os.environ["PATH"]
        return env

    def prompt_metadata(self, user_message, prompt):
        _, metadata = self._build_prompt_and_metadata(user_message)
        return metadata

    def _build_prompt_and_metadata(self, user_message):
        refresh = self.refresh_prefix()
        self.resume_state = self.evaluate_resume_state()
        prompt, metadata = self.context_manager.build(user_message)
        # 这里把“这轮 prompt 是怎么拼出来的”连同缓存相关状态一起记下来，
        # 后面 trace/report 才能解释清楚：为什么这一轮 prefix 变了、缓存有没有命中。
        metadata.update(
            {
                "prefix_chars": len(self.prefix),
                "workspace_chars": len(self.workspace.text()),
                "memory_chars": len(self.memory_text()),
                "history_chars": len(self.history_text()),
                "request_chars": len(user_message),
                "tool_count": len(self.tools),
                "workspace_docs": len(self.workspace.project_docs),
                "recent_commits": len(self.workspace.recent_commits),
                "prefix_hash": self.prefix_state.hash,
                "prompt_cache_key": self.prefix_state.hash,
                "workspace_fingerprint": self.prefix_state.workspace_fingerprint,
                "tool_signature": self.prefix_state.tool_signature,
                "workspace_changed": refresh["workspace_changed"],
                "prefix_changed": refresh["prefix_changed"],
                "prompt_cache_supported": bool(getattr(self.model_client, "supports_prompt_cache", False)),
                "resume_status": self.resume_state.get("status", CHECKPOINT_NONE_STATUS),
                "stale_summary_invalidations": int(self.resume_state.get("stale_summary_invalidations", 0)),
                "stale_paths": list(self.resume_state.get("stale_paths", [])),
                "runtime_identity_mismatch_fields": list(self.resume_state.get("runtime_identity_mismatch_fields", [])),
                "execution_mode": self.execution_mode,
            }
        )
        if self.execution_mode == ENHANCED_CONTRACT_MODE and self.active_contract is not None:
            metadata.update(
                {
                    "contract_version": self.active_contract.contract_version,
                    "contract_projection_digest": self.protected_contract_projection_digest(),
                }
            )
        metadata.update(self.detected_secret_env_summary())
        return prompt, metadata

    def emit_trace(self, task_state, event, payload=None):
        payload = self.redact_artifact(payload or {})
        payload.setdefault("task_id", task_state.task_id)
        if self.execution_mode:
            payload.setdefault("execution_mode", self.execution_mode)
        if self.active_contract is not None:
            payload.setdefault("contract_version", self.active_contract.contract_version)
        payload["event"] = event
        payload["created_at"] = now()
        # trace 是运行中的逐事件时间线，适合回答“这一轮 agent 到底做了什么”。
        self.run_store.append_trace(task_state, payload)
        return payload

    def capture_workspace_snapshot(self):
        snapshot = {}
        for path in self.root.rglob("*"):
            try:
                relative_parts = path.relative_to(self.root).parts
            except ValueError:
                continue
            if any(part in IGNORED_PATH_NAMES for part in relative_parts):
                continue
            if not path.is_file():
                continue
            try:
                snapshot[path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception:
                continue
        return snapshot

    @staticmethod
    def diff_workspace_snapshots(before, after):
        changed_paths = []
        summaries = []
        all_paths = sorted(set(before) | set(after))
        for path in all_paths:
            if before.get(path) == after.get(path):
                continue
            changed_paths.append(path)
            if path not in before:
                summaries.append(f"created:{path}")
            elif path not in after:
                summaries.append(f"deleted:{path}")
            else:
                summaries.append(f"modified:{path}")
        return changed_paths, summaries

    def create_checkpoint(self, task_state, user_message, trigger):
        state = self.checkpoint_state()
        current = self.current_checkpoint()
        checkpoint_id = "ckpt_" + uuid.uuid4().hex[:8]
        key_files = []
        freshness = {}
        for path in self.memory.to_dict()["working"]["recent_files"]:
            file_freshness = memorylib.file_freshness(path, self.root)
            freshness[path] = file_freshness
            key_files.append({"path": path, "freshness": file_freshness})
        checkpoint = {
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": current.get("checkpoint_id", "") if current else "",
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "created_at": now(),
            "current_goal": str(user_message),
            "execution_mode": task_state.execution_mode or self.execution_mode or COMPATIBILITY_MODE,
            "completion_assurance": task_state.completion_assurance,
            "contract_summary": dict(task_state.contract_summary),
            "completion_candidate": dict(task_state.completion_candidate),
            "current_evidence_summary": dict(task_state.current_evidence_summary),
            "condition_coverage": dict(task_state.condition_coverage),
            "completion_verdict": dict(task_state.completion_verdict),
            "completed": [task_state.final_answer]
            if task_state.completion_assurance == "verified" and task_state.final_answer
            else [],
            "excluded": [],
            "current_blocker": "" if str(task_state.stop_reason or "") in ("", "final_answer_returned") else str(task_state.stop_reason),
            "next_step": self.infer_next_step(task_state),
            "key_files": key_files,
            "freshness": freshness,
            "summary": f"{trigger}: {clip(str(user_message), 120)}",
            "runtime_identity": self.current_runtime_identity(),
        }
        state["items"][checkpoint_id] = checkpoint
        state["current_id"] = checkpoint_id
        task_state.checkpoint_id = checkpoint_id
        self.session["runtime_identity"] = checkpoint["runtime_identity"]
        self.session_path = self.session_store.save(self.session)
        return checkpoint

    def infer_next_step(self, task_state):
        if task_state.status == "completed":
            return "No next step recorded."
        if task_state.stop_reason == "step_limit_reached":
            return "Resume from the latest checkpoint and continue the task."
        if task_state.last_tool:
            return f"Decide the next action after {task_state.last_tool}."
        return "Continue the task from the latest checkpoint."

    def update_memory_after_tool(self, name, args, result):
        """把少量高价值工具结果沉淀到 working memory。

        为什么存在：
        并不是每个工具结果都值得长期带进下一轮 prompt。完整结果已经进了
        `history`，这里只挑少量“下一轮大概率还会用到”的事实做提纯，
        例如最近读写过哪些文件、某个文件读出来的短摘要。

        输入 / 输出：
        - 输入：工具名 `name`、参数 `args`、执行结果 `result`
        - 输出：无显式返回值，副作用是更新 `self.memory`

        在 agent 链路里的位置：
        它发生在 `run_tool()` 真正执行完工具之后、下一轮 prompt 组装之前。
        也就是说：工具结果先进入完整历史，再由这个函数择优沉淀成轻量记忆。
        """
        if not self.feature_enabled("memory"):
            return
        path = args.get("path")
        if not path:
            return

        canonical_path = self.memory.canonical_path(path)
        # 不是所有工具结果都进入工作记忆。
        # 读文件会生成摘要；写文件/patch 会让旧摘要失效，因为它们可能过期了。
        if name in {"read_file", "write_file", "patch_file"}:
            self.memory.remember_file(canonical_path)
        if name == "read_file":
            summary = memorylib.summarize_read_result(result)
            self.memory.set_file_summary(canonical_path, summary)
            self.memory.append_note(summary, tags=(canonical_path,), source=canonical_path)
        elif name in {"write_file", "patch_file"}:
            self.memory.invalidate_file_summary(canonical_path)

    def note_tool(self, name, args, result):
        self.update_memory_after_tool(name, args, result)

    def record_process_note_for_tool(self, name, metadata):
        status = str(metadata.get("tool_status", "")).strip()
        if status not in {"partial_success", "error", "rejected"}:
            return
        affected_paths = [str(path).strip() for path in metadata.get("affected_paths", []) if str(path).strip()]
        path_text = ", ".join(affected_paths) or "workspace"
        if status == "partial_success":
            text = f"{name} partial_success on {path_text}; inspect diff before retry"
        elif status == "error":
            text = f"{name} error on {path_text}; check the failure before retry"
        else:
            text = f"{name} rejected; choose a different action before retry"
        tags = ["process", status, *affected_paths]
        self.memory.append_note(text, tags=tuple(tags), source=name, kind="process")
        self.session["memory"] = self.memory.to_dict()

    def reject_durable_reason(self, note_text):
        text = str(note_text or "").strip()
        lowered = text.lower()
        if not text:
            return "empty"
        if REDACTED_VALUE in text or SECRET_SHAPED_TEXT_PATTERN.search(text):
            return "secret_shaped"
        checkpoint_like_prefixes = (
            "current goal",
            "current blocker",
            "next step",
            "current phase",
            "key files",
            "freshness",
            "当前目标",
            "当前卡点",
            "下一步",
            "当前阶段",
            "关键文件",
            "已完成",
            "已排除",
        )
        if any(lowered.startswith(prefix) for prefix in checkpoint_like_prefixes):
            return "transient_task_state"
        if re.search(r"(?i)\b(stdout|stderr|traceback|exit_code)\b", text) or len(text) > 220:
            return "noisy_output"
        return ""

    def extract_durable_promotions(self, user_message, final_answer):
        user_text = str(user_message or "")
        if not (DURABLE_MEMORY_INTENT_PATTERN.search(user_text) or DURABLE_MEMORY_INTENT_ZH_PATTERN.search(user_text)):
            return [], []
        promotions = []
        rejections = []
        for line in str(final_answer or "").splitlines():
            text = line.strip()
            if not text or REDACTED_VALUE in text:
                continue
            for topic, pattern in DURABLE_MEMORY_LINE_PATTERNS:
                match = pattern.match(text)
                if not match:
                    continue
                note_text = match.group(1).strip()
                if note_text:
                    reason = self.reject_durable_reason(note_text)
                    if reason:
                        rejections.append(f"{topic}:{reason}")
                        break
                    promotions.append((topic, note_text))
                break
        return promotions, rejections

    def promote_durable_memory(self, user_message, final_answer):
        promotions, rejections = self.extract_durable_promotions(user_message, final_answer)
        promoted, superseded = self.memory.promote_durable(promotions)
        self.session["memory"] = self.memory.to_dict()
        self.last_durable_promotions = promoted
        self.last_durable_rejections = rejections
        self.last_durable_superseded = superseded
        return promoted, rejections, superseded

    def protected_contract_text(self):
        contract = self.active_contract
        if self.execution_mode != ENHANCED_CONTRACT_MODE or contract is None:
            return ""
        condition = contract.completion_conditions[0]
        return "\n".join(
            [
                "Protected Current Task Contract:",
                f"contract_id: {contract.contract_id}",
                f"contract_version: {contract.contract_version}",
                f"goal: {contract.goal}",
                f"workspace_scope: {contract.workspace_scope}",
                f"target_path: {contract.target_path}",
                f"expected_content_basis: {condition.kind}",
                f"expected_content: {json.dumps(contract.expected_content, ensure_ascii=False)}",
                f"expected_sha256: {condition.expected_sha256}",
                f"expected_size: {condition.expected_size}",
                f"condition_id: {condition.condition_id}",
                f"allowed_tools: {', '.join(contract.allowed_tools)}",
                f"prohibitions: {', '.join(contract.prohibitions)}",
            ]
        )

    def protected_contract_projection_digest(self):
        text = self.protected_contract_text()
        return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""

    def _activate_contract(self, task_state, task_contract):
        if self.execution_mode == COMPATIBILITY_MODE:
            self.contract_activation = {
                "mode": COMPATIBILITY_MODE,
                "status": "not_requested",
            }
            self.emit_trace(
                task_state,
                "execution_mode_selected",
                {"execution_mode": COMPATIBILITY_MODE},
            )
            return None

        self.emit_trace(
            task_state,
            "execution_mode_selected",
            {"execution_mode": ENHANCED_CONTRACT_MODE},
        )
        try:
            contract = TaskContract.from_mapping(task_contract, task_id=task_state.task_id)
        except ContractValidationError as exc:
            failure = {
                "mode": ENHANCED_CONTRACT_MODE,
                "code": "contract_validation_failed",
                "validation_code": exc.code,
                "message": str(exc),
            }
            self.contract_activation = {
                "mode": ENHANCED_CONTRACT_MODE,
                "status": "failed",
                "failure_code": failure["code"],
                "validation_code": failure["validation_code"],
            }
            return failure
        except ValueError as exc:
            failure = {
                "mode": ENHANCED_CONTRACT_MODE,
                "code": "contract_validation_failed",
                "validation_code": _legacy_contract_validation_code(exc),
                "message": str(exc),
            }
            self.contract_activation = {
                "mode": ENHANCED_CONTRACT_MODE,
                "status": "failed",
                "failure_code": failure["code"],
                "validation_code": failure["validation_code"],
            }
            return failure

        if contract.execution_not_ready:
            failure = {
                "mode": ENHANCED_CONTRACT_MODE,
                "code": "contract_not_execution_ready",
                "reason_code": "contract_not_execution_ready",
                "execution_not_ready": True,
                "lifecycle_state": contract.lifecycle_state,
            }
            self.contract_activation = {
                "mode": ENHANCED_CONTRACT_MODE,
                "status": "failed",
                "failure_code": failure["code"],
                "execution_not_ready": True,
                "lifecycle_state": contract.lifecycle_state,
            }
            return failure

        self.active_contract = contract
        self.evidence_ledger = EvidenceLedger(contract)
        task_state.contract_summary = contract.summary()
        self.contract_activation = {
            "mode": ENHANCED_CONTRACT_MODE,
            "status": "active",
            "contract_id": contract.contract_id,
            "contract_version": contract.contract_version,
            "task_id": contract.task_id,
            "lifecycle_state": contract.lifecycle_state,
            "execution_not_ready": contract.execution_not_ready,
            "projection_digest": self.protected_contract_projection_digest(),
        }
        self.emit_trace(
            task_state,
            "contract_activated",
            {
                "execution_mode": ENHANCED_CONTRACT_MODE,
                "task_id": task_state.task_id,
                "contract_id": contract.contract_id,
                "contract_version": contract.contract_version,
                "contract": contract.summary(),
                "projection_digest": self.contract_activation["projection_digest"],
            },
        )
        return None

    def _finish_pre_model_failure(self, task_state, failure, result_prefix):
        self.last_activation_failure = dict(failure)
        if self.execution_mode == ENHANCED_CONTRACT_MODE:
            task_state.completion_assurance = "enhanced_non_success"
        if failure.get("code") == "contract_not_execution_ready":
            task_state.completion_verdict = {
                "result": "contract_blocked",
                "candidate_id": "",
                "covered_condition_ids": [],
                "unmet_condition_ids": [],
                "evidence_ids": [],
                "reason_code": "contract_not_execution_ready",
            }
        task_state.stop("contract_activation_failed", status="failed")
        self.run_store.write_task_state(task_state)
        event = "contract_projection_failed" if result_prefix == "contract_projection_failed" else "contract_activation_failed"
        self.emit_trace(task_state, event, dict(failure))
        self.emit_trace(
            task_state,
            "run_finished",
            {
                "status": task_state.status,
                "stop_reason": task_state.stop_reason,
                "execution_mode": self.execution_mode,
            },
        )
        self.run_store.write_report(task_state, self.redact_artifact(self.build_report(task_state)))
        code = str(failure.get("code", "unknown_failure"))
        return f"{result_prefix}:{code}"

    def ask(self, user_message, *, task_contract=None):
        """执行一次完整的 agent 回合，直到产出最终答案或命中停止条件。

        为什么存在：
        `ask()` 是整个 runtime 的总调度器。它把“用户提一个请求”扩展成一条
        可持续推进的控制循环：记录会话、组 prompt、调用模型、执行工具、
        写 trace/report、更新状态，直到模型给出最终答案或系统主动停下。

        输入 / 输出：
        - 输入：`user_message`，即用户这一次的任务描述
        - 输出：字符串形式的最终回答；如果中途达到步数上限或重试上限，
          返回的是一条停止原因说明

        在 agent 链路里的位置：
        它是 CLI 和底层工具/模型之间的核心桥梁。CLI 收到用户输入后基本只做
        一件事：调用 `agent.ask()`。而 `ask()` 内部再去驱动 `ContextManager`
        组 prompt、`model_client.complete()` 调模型、`run_tool()` 执行动作。
        如果新人想理解 ForgePilot 是怎么“从一句话跑成一个 agent 流程”的，
        这里就是最关键的入口。
        """
        run_started_at = time.monotonic()
        self.execution_mode = (
            ENHANCED_CONTRACT_MODE if task_contract is not None else COMPATIBILITY_MODE
        )
        self.active_contract = None
        self.evidence_ledger = None
        self._completion_candidate_sequence = 0
        self._observation_sequence = 0
        self.contract_activation = {"mode": self.execution_mode, "status": "pending"}
        self.last_activation_failure = {}
        self.memory.set_task_summary(user_message)
        self.record({"role": "user", "content": user_message, "created_at": now()})

        task_state = TaskState.create(run_id=self.new_run_id(), task_id=self.new_task_id(), user_request=user_message)
        task_state.resume_status = self.resume_state.get("status", CHECKPOINT_NONE_STATUS)
        task_state.execution_mode = self.execution_mode
        task_state.completion_assurance = (
            "legacy_unverified" if self.execution_mode == COMPATIBILITY_MODE else "enhanced_unverified"
        )
        self.current_task_state = task_state
        self.current_run_dir = self.run_store.start_run(task_state)
        self.emit_trace(
            task_state,
            "run_started",
            {
                "task_id": task_state.task_id,
                "user_request": clip(user_message, 300),
                "execution_mode": self.execution_mode,
            },
        )

        activation_failure = self._activate_contract(task_state, task_contract)
        if activation_failure is not None:
            return self._finish_pre_model_failure(
                task_state, activation_failure, "contract_activation_failed"
            )

        tool_steps = 0
        attempts = 0
        max_attempts = max(self.max_steps * 3, self.max_steps + 4)

        # 这是 agent 的主循环，可以按“感知 -> 决策 -> 行动 -> 记录”来理解：
        # 1. 感知：重新组 prompt，把当前状态整理给模型看
        # 2. 决策：让模型返回一个工具调用，或一个最终答案
        # 3. 行动：如果是工具调用，就执行工具
        # 4. 记录：把结果写回 history / task_state / trace / memory
        # 然后进入下一轮，直到停机条件满足
        while tool_steps < self.max_steps and attempts < max_attempts:
            attempts += 1
            task_state.record_attempt()
            self.run_store.write_task_state(task_state)
            prompt_started_at = time.monotonic()
            try:
                prompt, prompt_metadata = self._build_prompt_and_metadata(user_message)
            except ProtectedContractBudgetError as exc:
                failure = {
                    "mode": self.execution_mode,
                    "code": exc.code,
                    "contract_version": getattr(self.active_contract, "contract_version", ""),
                    "projection_digest": self.protected_contract_projection_digest(),
                    "protected_chars": exc.protected_chars,
                    "request_chars": exc.request_chars,
                    "total_budget": exc.total_budget,
                }
                return self._finish_pre_model_failure(
                    task_state, failure, "contract_projection_failed"
                )
            self.emit_trace(
                task_state,
                "prompt_built",
                {
                    "prompt_metadata": prompt_metadata,
                    "duration_ms": int((time.monotonic() - prompt_started_at) * 1000),
                },
            )
            if prompt_metadata.get("resume_status") == CHECKPOINT_PARTIAL_STALE_STATUS:
                checkpoint = self.create_checkpoint(task_state, user_message, trigger="freshness_mismatch")
                self.run_store.write_task_state(task_state)
                self.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "freshness_mismatch",
                    },
                )
            elif prompt_metadata.get("resume_status") == CHECKPOINT_WORKSPACE_MISMATCH_STATUS:
                self.emit_trace(
                    task_state,
                    "runtime_identity_mismatch",
                    {
                        "fields": list(prompt_metadata.get("runtime_identity_mismatch_fields", [])),
                    },
                )
                checkpoint = self.create_checkpoint(task_state, user_message, trigger="workspace_mismatch")
                self.run_store.write_task_state(task_state)
                self.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "workspace_mismatch",
                    },
                )
            if prompt_metadata.get("budget_reductions"):
                checkpoint = self.create_checkpoint(task_state, user_message, trigger="context_reduction")
                self.run_store.write_task_state(task_state)
                self.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "context_reduction",
                    },
                )
            self.emit_trace(
                task_state,
                "model_requested",
                {
                    "attempts": task_state.attempts,
                    "tool_steps": task_state.tool_steps,
                    "prompt_cache_key": prompt_metadata.get("prompt_cache_key"),
                },
            )
            prompt_cache_key = None
            prompt_cache_retention = None
            if getattr(self.model_client, "supports_prompt_cache", False):
                # 只有后端明确支持时，才把稳定前缀的 hash 作为 cache key 发出去。
                prompt_cache_key = prompt_metadata.get("prompt_cache_key")
                prompt_cache_retention = "in_memory"
            model_started_at = time.monotonic()
            try:
                raw = self.model_client.complete(
                    prompt,
                    self.max_new_tokens,
                    prompt_cache_key=prompt_cache_key,
                    prompt_cache_retention=prompt_cache_retention,
                )
            except ModelExhaustedError:
                if (
                    self.execution_mode == ENHANCED_CONTRACT_MODE
                    and task_state.completion_candidate
                ):
                    task_state.completion_assurance = "enhanced_non_success"
                    task_state.stop("evidence_insufficient", status="stopped")
                    fallback_candidate = str(task_state.completion_candidate.get("text", ""))
                    self.record({"role": "completion_candidate", "content": fallback_candidate, "created_at": now()})
                    self.run_store.write_task_state(task_state)
                    checkpoint = self.create_checkpoint(
                        task_state, user_message, trigger="evidence_insufficient"
                    )
                    self.emit_trace(
                        task_state,
                        "run_finished",
                        {
                            "status": task_state.status,
                            "stop_reason": task_state.stop_reason,
                            "execution_mode": self.execution_mode,
                        },
                    )
                    self.run_store.write_report(
                        task_state, self.redact_artifact(self.build_report(task_state))
                    )
                    return fallback_candidate
                task_state.completion_assurance = (
                    "legacy_unverified"
                    if self.execution_mode == COMPATIBILITY_MODE
                    else "enhanced_non_success"
                )
                task_state.stop_model_error()
                self.run_store.write_task_state(task_state)
                self.emit_trace(
                    task_state,
                    "run_failed",
                    {
                        "status": task_state.status,
                        "stop_reason": task_state.stop_reason,
                        "execution_mode": self.execution_mode,
                    },
                )
                self.run_store.write_report(
                    task_state, self.redact_artifact(self.build_report(task_state))
                )
                raise
            except Exception:
                task_state.completion_assurance = (
                    "legacy_unverified"
                    if self.execution_mode == COMPATIBILITY_MODE
                    else "enhanced_non_success"
                )
                task_state.stop_model_error()
                self.run_store.write_task_state(task_state)
                self.emit_trace(
                    task_state,
                    "run_failed",
                    {
                        "status": task_state.status,
                        "stop_reason": task_state.stop_reason,
                        "execution_mode": self.execution_mode,
                    },
                )
                self.run_store.write_report(
                    task_state, self.redact_artifact(self.build_report(task_state))
                )
                raise
            completion_metadata = dict(getattr(self.model_client, "last_completion_metadata", {}) or {})
            if completion_metadata:
                # 把后端返回的 usage/cache 统计并回 prompt_metadata，
                # 方便统一写入 report 和 trace。
                prompt_metadata.update(completion_metadata)
            self.last_completion_metadata = completion_metadata
            self.last_prompt_metadata = prompt_metadata
            kind, payload = self.parse(raw)
            self.emit_trace(
                task_state,
                "model_parsed",
                {
                    "kind": kind,
                    "completion_metadata": completion_metadata,
                    "duration_ms": int((time.monotonic() - model_started_at) * 1000),
                },
            )

            if kind == "tool":
                tool_steps += 1
                name = payload.get("name", "")
                args = payload.get("args", {})
                task_state.record_tool(name)
                tool_started_at = time.monotonic()
                result = self.run_tool(name, args)
                self.record(
                    {
                        "role": "tool",
                        "name": name,
                        "args": args,
                        "content": result,
                        "created_at": now(),
                    }
                )
                self.run_store.write_task_state(task_state)
                tool_metadata = dict(self._last_tool_result_metadata or {})
                if tool_metadata.get("admission_decision") == "blocked":
                    tool_event = "task_admission_blocked"
                elif not tool_metadata.get("execution_attempted", False):
                    tool_event = "tool_safety_rejected"
                else:
                    tool_event = "tool_executed"
                self.emit_trace(
                    task_state,
                    tool_event,
                    {
                        "name": name,
                        "args": args,
                        "result": clip(result, 500),
                        "duration_ms": int((time.monotonic() - tool_started_at) * 1000),
                        **tool_metadata,
                    },
                )
                checkpoint = self.create_checkpoint(task_state, user_message, trigger=tool_event)
                self.run_store.write_task_state(task_state)
                self.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": tool_event,
                    },
                )
                continue

            if kind == "retry":
                self.record({"role": "assistant", "content": payload, "created_at": now()})
                self.run_store.write_task_state(task_state)
                continue

            final = (payload or raw).strip()
            if self.execution_mode == ENHANCED_CONTRACT_MODE:
                candidate = CompletionCandidate(
                    candidate_id=self._next_completion_candidate_id(),
                    text=final,
                )
                task_state.completion_candidate = {
                    "candidate_id": candidate.candidate_id,
                    "text": candidate.text,
                }
                self.emit_trace(
                    task_state,
                    "completion_candidate_received",
                    {
                        "execution_mode": self.execution_mode,
                        "task_id": task_state.task_id,
                        "contract_version": self.active_contract.contract_version,
                        "candidate_id": candidate.candidate_id,
                    },
                )
                verdict = evaluate_completion(self.active_contract, self.evidence_ledger, candidate)
                verdict_data = {
                    "result": verdict.result,
                    "candidate_id": verdict.candidate_id,
                    "covered_condition_ids": list(verdict.covered_condition_ids),
                    "unmet_condition_ids": list(verdict.unmet_condition_ids),
                    "evidence_ids": list(verdict.evidence_ids),
                    "reason_code": verdict.reason_code,
                }
                task_state.completion_verdict = verdict_data
                task_state.completion_assurance = (
                    "verified" if verdict.result == "verified_completed" else "enhanced_non_success"
                )
                self.emit_trace(task_state, "completion_coverage_evaluated", {
                    "execution_mode": self.execution_mode,
                    "task_id": task_state.task_id,
                    "contract_version": self.active_contract.contract_version,
                    "candidate_id": verdict.candidate_id,
                    "covered_condition_ids": list(verdict.covered_condition_ids),
                    "unmet_condition_ids": list(verdict.unmet_condition_ids),
                    "evidence_ids": list(verdict.evidence_ids),
                })
                self.emit_trace(
                    task_state,
                    "completion_verdict_decided",
                    {
                        "execution_mode": self.execution_mode,
                        "task_id": task_state.task_id,
                        "contract_version": self.active_contract.contract_version,
                        **verdict_data,
                    },
                )
                if verdict.result != "verified_completed":
                    unmet = ", ".join(verdict.unmet_condition_ids) or "none"
                    feedback = (
                        f"Completion verdict: {verdict.result}; unmet conditions: {unmet}. "
                        "Obtain current verified Evidence before claiming completion."
                    )
                    self.record({
                        "role": "completion_feedback",
                        "content": feedback,
                        "created_at": now(),
                    })
                    self.run_store.write_task_state(task_state)
                    continue
            else:
                task_state.completion_assurance = "legacy_unverified"
            self.record({"role": "assistant", "content": final, "created_at": now()})
            task_state.finish_success(final)
            self.promote_durable_memory(user_message, final)
            checkpoint = self.create_checkpoint(task_state, user_message, trigger="run_finished")
            self.run_store.write_task_state(task_state)
            self.emit_trace(
                task_state,
                "checkpoint_created",
                {
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "trigger": "run_finished",
                },
            )
            self.emit_trace(
                task_state,
                "run_finished",
                {
                    "status": task_state.status,
                    "stop_reason": task_state.stop_reason,
                    "final_answer": final,
                    "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
                },
            )
            self.run_store.write_report(task_state, self.redact_artifact(self.build_report(task_state)))
            return final

        enhanced_non_success = self.execution_mode == ENHANCED_CONTRACT_MODE
        if enhanced_non_success and task_state.completion_verdict:
            result_code = str(task_state.completion_verdict.get("result", "evidence_insufficient"))
            unmet = task_state.completion_verdict.get("unmet_condition_ids", [])
            final = f"{result_code}:{','.join(str(item) for item in unmet) or 'none'}"
            task_state.completion_assurance = "enhanced_non_success"
            task_state.stop(result_code, status="stopped")
        elif attempts >= max_attempts and tool_steps < self.max_steps:
            final = "Stopped after too many malformed model responses without a valid tool call or final answer."
            if enhanced_non_success:
                task_state.completion_assurance = "enhanced_non_success"
                task_state.stop_retry_limit()
            else:
                task_state.stop_retry_limit(final)
        else:
            final = "Stopped after reaching the step limit without a final answer."
            if enhanced_non_success:
                task_state.completion_assurance = "enhanced_non_success"
                task_state.stop_step_limit()
            else:
                task_state.stop_step_limit(final)
        self.record({"role": "stop_notice" if enhanced_non_success else "assistant", "content": final, "created_at": now()})
        if not enhanced_non_success:
            self.promote_durable_memory(user_message, final)
        self.run_store.write_task_state(task_state)
        checkpoint = self.create_checkpoint(task_state, user_message, trigger=task_state.stop_reason or "run_stopped")
        self.emit_trace(
            task_state,
            "checkpoint_created",
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "trigger": task_state.stop_reason or "run_stopped",
            },
        )
        self.emit_trace(
            task_state,
            "run_finished",
            {
                "status": task_state.status,
                "stop_reason": task_state.stop_reason,
                "final_answer": final,
                "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
            },
        )
        self.run_store.write_report(task_state, self.redact_artifact(self.build_report(task_state)))
        return final

    def _next_action_candidate_id(self):
        self._action_candidate_sequence += 1
        return f"candidate-{self._action_candidate_sequence:04d}"

    def _next_completion_candidate_id(self):
        self._completion_candidate_sequence += 1
        return f"completion-{self._completion_candidate_sequence:04d}"

    def _capture_contract_target_state(self, target_path):
        try:
            path = self.path(target_path)
            if not path.is_file():
                return {"exists": False, "sha256": "", "size": 0}
            content = path.read_bytes()
            return {
                "exists": True,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
            }
        except (OSError, ValueError):
            return {"exists": False, "sha256": "", "size": 0}

    @staticmethod
    def _evidence_record_summary(record):
        if record is None:
            return {}
        return {
            "evidence_id": record.evidence_id,
            "observation_id": record.observation_id,
            "condition_id": record.condition_id,
            "eligible": record.eligible,
            "verified": record.verified,
            "match": record.match,
            "freshness": record.freshness,
            "target_revision": record.target_revision,
            "reason_code": record.reason_code,
            "resolves_observation_ids": list(record.resolves_observation_ids),
        }

    def _refresh_task_evidence_summary(self, record=None):
        if self.current_task_state is None or self.evidence_ledger is None:
            return
        current = self.evidence_ledger.current_evidence()
        self.current_task_state.current_evidence_summary = self._evidence_record_summary(current)
        self.current_task_state.condition_coverage = {
            self.active_contract.completion_conditions[0].condition_id: {
                "covered": current is not None,
                "evidence_id": current.evidence_id if current else "",
                "target_revision": current.target_revision if current else self.evidence_ledger.target_revision,
                "unmet_reason": "" if current else "current_evidence_insufficient",
            }
        }

    def _record_enhanced_observation(self, candidate, before_state, after_state):
        if self.evidence_ledger is None or self.active_contract is None:
            return None
        if candidate.tool_name not in {"read_file", "write_file", "patch_file"}:
            return None
        if (
            candidate.tool_name == "read_file"
            and candidate.arguments.get("path") != self.active_contract.target_path
        ):
            return None
        metadata = dict(self._last_tool_result_metadata or {})
        if not metadata.get("execution_attempted"):
            return None
        status = {
            "ok": "success",
            "partial_success": "partial",
            "error": "error",
        }.get(str(metadata.get("tool_status", "")), str(metadata.get("tool_status", "error")))
        self._observation_sequence += 1
        observation = ToolObservation(
            observation_id=f"observation-{self._observation_sequence:04d}",
            task_id=self.active_contract.task_id,
            contract_version=self.active_contract.contract_version,
            candidate_id=candidate.candidate_id,
            tool_name=candidate.tool_name,
            status=status,
            execution_attempted=True,
            target_path=self.active_contract.target_path,
            target_exists=after_state["exists"],
            target_sha256=after_state["sha256"],
            target_size=after_state["size"],
            target_revision=self.evidence_ledger.target_revision,
            target_before_exists=before_state["exists"],
            target_before_sha256=before_state["sha256"],
            target_before_size=before_state["size"],
        )
        previous_revision = self.evidence_ledger.target_revision
        record = self.evidence_ledger.record(observation)
        metadata["observation_id"] = observation.observation_id
        metadata["evidence_id"] = record.evidence_id
        metadata["target_revision"] = record.target_revision
        self._last_tool_result_metadata.update(metadata)
        self._refresh_task_evidence_summary(record)
        if self.current_task_state is not None:
            self.emit_trace(
                self.current_task_state,
                "environment_observed",
                {
                    "execution_mode": self.execution_mode,
                    "task_id": observation.task_id,
                    "observation_id": observation.observation_id,
                    "candidate_id": observation.candidate_id,
                    "contract_version": observation.contract_version,
                    "target_path": observation.target_path,
                    "target_before_exists": observation.target_before_exists,
                    "target_before_sha256": observation.target_before_sha256,
                    "target_before_size": observation.target_before_size,
                    "target_exists": observation.target_exists,
                    "target_sha256": observation.target_sha256,
                    "target_size": observation.target_size,
                    "target_revision": record.target_revision,
                    "status": observation.status,
                },
            )
            self.emit_trace(
                self.current_task_state,
                "evidence_recorded",
                {
                    "execution_mode": self.execution_mode,
                    "task_id": self.current_task_state.task_id,
                    "contract_version": self.active_contract.contract_version,
                    "evidence_id": record.evidence_id,
                    "observation_id": record.observation_id,
                    "condition_id": record.condition_id,
                    "match": record.match,
                    "eligible": record.eligible,
                    "verified": record.verified,
                    "freshness": record.freshness,
                    "target_revision": record.target_revision,
                    "reason_code": record.reason_code,
                },
            )
            if record.target_revision != previous_revision or record.resolves_observation_ids:
                self.emit_trace(
                    self.current_task_state,
                    "evidence_superseded_or_resolved",
                    {
                        "execution_mode": self.execution_mode,
                        "task_id": self.current_task_state.task_id,
                        "contract_version": self.active_contract.contract_version,
                        "evidence_id": record.evidence_id,
                        "observation_id": record.observation_id,
                        "target_revision": record.target_revision,
                        "resolves_observation_ids": list(record.resolves_observation_ids),
                    },
                )
        return record

    def _reject_task_admission(self, name, reason_code, constraint_id="", normalized_path=""):
        tool = self.tools.get(name)
        self._last_tool_result_metadata = {
            "tool_status": "rejected",
            "tool_error_code": f"task_admission_{reason_code}",
            "admission_decision": "blocked",
            "admission_reason_code": reason_code,
            "admission_constraint_id": constraint_id,
            "normalized_path": normalized_path,
            "execution_attempted": False,
            "execution_mode": self.execution_mode,
            "task_id": self.current_task_state.task_id if self.current_task_state is not None else "",
            "contract_version": getattr(self.active_contract, "contract_version", ""),
            "security_event_type": "",
            "risk_level": "high" if tool and tool["risky"] else "low",
            "read_only": bool(tool) and not tool["risky"],
            "affected_paths": [],
            "workspace_changed": False,
            "diff_summary": [],
        }
        if self.current_task_state is not None:
            self.emit_trace(
                self.current_task_state,
                "task_admission_decided",
                {
                    "execution_mode": self.execution_mode,
                    "task_id": self.current_task_state.task_id,
                    "contract_version": getattr(self.active_contract, "contract_version", ""),
                    "tool_name": name,
                    "admission_decision": "blocked",
                    "reason_code": reason_code,
                    "constraint_id": constraint_id,
                    "normalized_path": normalized_path,
                },
            )
        return f"error: task admission blocked: {reason_code}"

    def run_tool(self, name, args):
        """Route a tool candidate through the frozen execution-mode gateway."""
        args = args or {}
        candidate = None
        before_state = None
        if self.execution_mode == ENHANCED_CONTRACT_MODE:
            contract = self.active_contract
            if contract is None or contract.execution_not_ready:
                return self._reject_task_admission(
                    name, "contract_not_execution_ready", "execution_ready"
                )
            candidate = ActionCandidate(
                candidate_id=self._next_action_candidate_id(),
                tool_name=name,
                arguments=args,
            )
            decision = admit_action(contract, candidate)
            if not decision.admitted:
                return self._reject_task_admission(
                    name,
                    decision.reason_code,
                    decision.constraint_id,
                    decision.normalized_path,
                )
            before_state = self._capture_contract_target_state(contract.target_path)
            if self.current_task_state is not None:
                self.emit_trace(
                    self.current_task_state,
                    "task_admission_decided",
                    {
                        "execution_mode": self.execution_mode,
                        "task_id": self.current_task_state.task_id,
                        "contract_version": contract.contract_version,
                        "tool_name": name,
                        "candidate_id": candidate.candidate_id,
                        "admission_decision": "admitted",
                        "reason_code": "admitted",
                    },
                )
        result = self._run_tool_with_existing_safety(name, args)
        metadata = self._last_tool_result_metadata
        metadata.setdefault(
            "execution_attempted",
            metadata.get("tool_status") in {"ok", "partial_success", "error"},
        )
        if candidate is not None:
            metadata.update(
                {
                    "admission_decision": "admitted",
                    "admission_reason_code": "admitted",
                    "admission_candidate_id": candidate.candidate_id,
                    "execution_mode": self.execution_mode,
                    "task_id": self.current_task_state.task_id if self.current_task_state is not None else "",
                    "contract_version": self.active_contract.contract_version,
                }
            )
            if self.current_task_state is not None:
                self.emit_trace(
                    self.current_task_state,
                    "tool_safety_decided",
                    {
                        "execution_mode": self.execution_mode,
                        "task_id": self.current_task_state.task_id,
                        "contract_version": self.active_contract.contract_version,
                        "tool_name": name,
                        "candidate_id": candidate.candidate_id,
                        "execution_attempted": bool(metadata.get("execution_attempted")),
                        "tool_status": metadata.get("tool_status", ""),
                        "tool_error_code": metadata.get("tool_error_code", ""),
                    },
                )
            after_state = self._capture_contract_target_state(self.active_contract.target_path)
            self._record_enhanced_observation(candidate, before_state, after_state)
        return result

    def _run_tool_with_existing_safety(self, name, args):
        """执行一次工具调用，并在执行前后套上完整护栏。

        为什么存在：
        在 agent 系统里，真正危险的不是“模型会不会想调用工具”，而是
        “平台有没有在执行前把边界守住”。这个函数就是工具层的总闸口：
        所有工具调用都必须先经过它，不能让模型直接碰到底层函数。

        输入 / 输出：
        - 输入：工具名 `name`，参数字典 `args`
        - 输出：字符串结果。无论是成功结果还是错误信息，都会统一返回文本，
          这样模型下一轮都能继续消费这份反馈。

        在 agent 链路里的位置：
        它位于 `ask()` 的“模型决定要调用工具”之后，是控制循环里真正把模型
        意图落到外部世界的一步。因此这里串起了几乎所有安全与可控设计：
        工具是否存在、参数是否合法、是否重复、是否需要审批、执行结果是否裁剪、
        是否需要回写记忆。
        """
        # 工具执行不是“直接调函数”，而是一条带护栏的流水线：
        # 工具是否存在 -> 参数是否合法 -> 是否重复调用 -> 是否通过审批
        # -> 真正执行 -> 更新记忆。
        tool = self.tools.get(name)
        if tool is None:
            self._last_tool_result_metadata = {
                "tool_status": "rejected",
                "tool_error_code": "unknown_tool",
                "security_event_type": "",
                "risk_level": "high",
                "read_only": False,
                "affected_paths": [],
                "workspace_changed": False,
                "diff_summary": [],
            }
            return f"error: unknown tool '{name}'"
        try:
            self.validate_tool(name, args)
        except Exception as exc:
            example = self.tool_example(name)
            message = f"error: invalid arguments for {name}: {exc}"
            if example:
                message += f"\nexample: {example}"
            security_event_type = "path_escape" if "path escapes workspace" in str(exc) else ""
            self._last_tool_result_metadata = {
                "tool_status": "rejected",
                "tool_error_code": "invalid_arguments",
                "security_event_type": security_event_type,
                "risk_level": "high" if tool["risky"] else "low",
                "read_only": not tool["risky"],
                "affected_paths": [],
                "workspace_changed": False,
                "diff_summary": [],
            }
            return message
        if self.repeated_tool_call(name, args):
            self._last_tool_result_metadata = {
                "tool_status": "rejected",
                "tool_error_code": "repeated_identical_call",
                "security_event_type": "",
                "risk_level": "high" if tool["risky"] else "low",
                "read_only": not tool["risky"],
                "affected_paths": [],
                "workspace_changed": False,
                "diff_summary": [],
            }
            return f"error: repeated identical tool call for {name}; choose a different tool or return a final answer"
        if tool["risky"] and not self.approve(name, args):
            self._last_tool_result_metadata = {
                "tool_status": "rejected",
                "tool_error_code": "approval_denied",
                "security_event_type": "read_only_block" if self.read_only else "approval_denied",
                "risk_level": "high",
                "read_only": False,
                "affected_paths": [],
                "workspace_changed": False,
                "diff_summary": [],
            }
            return f"error: approval denied for {name}"
        before_snapshot = self.capture_workspace_snapshot() if tool["risky"] else {}
        after_snapshot = before_snapshot
        try:
            result = clip(tool["run"](args))
            after_snapshot = self.capture_workspace_snapshot() if tool["risky"] else before_snapshot
            affected_paths, diff_summary = self.diff_workspace_snapshots(before_snapshot, after_snapshot)
            workspace_changed = bool(affected_paths)
            tool_status = "ok"
            tool_error_code = ""
            if name == "run_shell":
                match = re.search(r"exit_code:\s*(-?\d+)", result)
                exit_code = int(match.group(1)) if match else 0
                if exit_code != 0 and workspace_changed:
                    tool_status = "partial_success"
                    tool_error_code = "tool_partial_success"
                elif exit_code != 0:
                    tool_status = "error"
                    tool_error_code = "tool_failed"
            self.update_memory_after_tool(name, args, result)
            self._last_tool_result_metadata = {
                "tool_status": tool_status,
                "tool_error_code": tool_error_code,
                "security_event_type": "",
                "risk_level": "high" if tool["risky"] else "low",
                "read_only": not tool["risky"],
                "affected_paths": affected_paths,
                "workspace_changed": workspace_changed,
                "workspace_fingerprint": self.workspace.fingerprint(),
                "diff_summary": diff_summary,
            }
            self.record_process_note_for_tool(name, self._last_tool_result_metadata)
            return result
        except Exception as exc:
            after_snapshot = self.capture_workspace_snapshot() if tool["risky"] else before_snapshot
            affected_paths, diff_summary = self.diff_workspace_snapshots(before_snapshot, after_snapshot)
            workspace_changed = bool(affected_paths)
            security_event_type = "path_escape" if "path escapes workspace" in str(exc) else ""
            self._last_tool_result_metadata = {
                "tool_status": "partial_success" if workspace_changed else "error",
                "tool_error_code": "tool_partial_success" if workspace_changed else "tool_failed",
                "security_event_type": security_event_type,
                "risk_level": "high" if tool["risky"] else "low",
                "read_only": not tool["risky"],
                "affected_paths": affected_paths,
                "workspace_changed": workspace_changed,
                "workspace_fingerprint": self.workspace.fingerprint(),
                "diff_summary": diff_summary,
            }
            self.record_process_note_for_tool(name, self._last_tool_result_metadata)
            return f"error: tool {name} failed: {exc}"

    def repeated_tool_call(self, name, args):
        # agent 很常见的一种坏循环，是在没有新信息的情况下反复发起同一调用。
        # 这里提前挡掉最简单的这种循环。
        tool_events = [item for item in self.session["history"] if item["role"] == "tool"]
        if len(tool_events) < 2:
            return False
        recent = tool_events[-2:]
        return all(item["name"] == name and item["args"] == args for item in recent)

    @staticmethod
    def new_task_id():
        return "task_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    @staticmethod
    def new_run_id():
        return "run_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    def build_report(self, task_state):
        # report 是一次运行的最终摘要；
        # 和 trace 的区别在于，trace 关注过程，report 关注结果与关键指标。
        return {
            "run_id": task_state.run_id,
            "task_id": task_state.task_id,
            "status": task_state.status,
            "stop_reason": task_state.stop_reason,
            "final_answer": task_state.final_answer,
            "tool_steps": task_state.tool_steps,
            "attempts": task_state.attempts,
            "checkpoint_id": task_state.checkpoint_id,
            "resume_status": task_state.resume_status,
            "execution_mode": self.execution_mode,
            "completion_assurance": task_state.completion_assurance,
            "contract_summary": dict(task_state.contract_summary),
            "completion_candidate": dict(task_state.completion_candidate),
            "current_evidence_summary": dict(task_state.current_evidence_summary),
            "condition_coverage": dict(task_state.condition_coverage),
            "completion_verdict": dict(task_state.completion_verdict),
            "contract_activation": dict(self.contract_activation),
            "activation_failure": dict(self.last_activation_failure),
            "task_state": task_state.to_dict(),
            "prompt_metadata": self.last_prompt_metadata,
            "durable_promotions": list(self.last_durable_promotions),
            "durable_rejections": list(self.last_durable_rejections),
            "durable_superseded": list(self.last_durable_superseded),
            "redacted_env": self.detected_secret_env_summary(),
            "plan": planner.plan_summary(self.session.get("plan")),
            "plan_items": planner.normalize_plan_state(self.session.get("plan"))["items"],
        }

    def tool_example(self, name):
        return toolkit.tool_example(name)

    def validate_tool(self, name, args):
        """把通用工具校验和 runtime 级额外约束串起来。"""
        toolkit.validate_tool(self, name, args)
        if name == "delegate":
            if self.depth >= self.max_depth:
                raise ValueError("delegate depth exceeded")

    def tool_list_files(self, args):
        return toolkit.tool_list_files(self, args)

    def tool_read_file(self, args):
        return toolkit.tool_read_file(self, args)

    def tool_search(self, args):
        return toolkit.tool_search(self, args)

    def tool_run_shell(self, args):
        return toolkit.tool_run_shell(self, args)

    def tool_write_file(self, args):
        return toolkit.tool_write_file(self, args)

    def tool_patch_file(self, args):
        return toolkit.tool_patch_file(self, args)

    def tool_delegate(self, args):
        return toolkit.tool_delegate(self, args)

    def approve(self, name, args):
        if self.read_only:
            return False
        if self.approval_policy == "auto":
            return True
        if self.approval_policy == "never":
            return False
        try:
            answer = input(f"approve {name} {json.dumps(args, ensure_ascii=True)}? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}

    @staticmethod
    def parse(raw):
        """把模型原始输出解析成 runtime 可执行的动作或最终答案。

        为什么存在：
        模型输出首先是自然语言文本，而 runtime 需要的是结构化决策：
        “这是工具调用”还是“这是最终答案”。如果没有这层解析，后面的工具校验、
        审批和执行链路就没法可靠工作。

        输入 / 输出：
        - 输入：模型返回的原始文本 `raw`
        - 输出：`(kind, payload)`，其中 `kind` 可能是 `tool`、`final`、`retry`

        在 agent 链路里的位置：
        它位于 `model_client.complete()` 之后、`run_tool()` 之前，是模型输出
        进入平台控制流的第一道结构化关口。
        """
        raw = str(raw)
        # 这里支持两种工具格式：
        # 1. <tool>...</tool> 里包 JSON，适合简短调用
        # 2. XML 风格属性/子标签，适合写文件这类多行内容
        if "<tool>" in raw and ("<final>" not in raw or raw.find("<tool>") < raw.find("<final>")):
            body = ForgePilot.extract(raw, "tool")
            try:
                payload = json.loads(body)
            except Exception:
                return "retry", ForgePilot.retry_notice("model returned malformed tool JSON")
            if not isinstance(payload, dict):
                return "retry", ForgePilot.retry_notice("tool payload must be a JSON object")
            if not str(payload.get("name", "")).strip():
                return "retry", ForgePilot.retry_notice("tool payload is missing a tool name")
            args = payload.get("args", {})
            if args is None:
                payload["args"] = {}
            elif not isinstance(args, dict):
                return "retry", ForgePilot.retry_notice()
            return "tool", payload
        if "<tool" in raw and ("<final>" not in raw or raw.find("<tool") < raw.find("<final>")):
            payload = ForgePilot.parse_xml_tool(raw)
            if payload is not None:
                return "tool", payload
            return "retry", ForgePilot.retry_notice()
        if "<final>" in raw:
            final = ForgePilot.extract(raw, "final").strip()
            if final:
                return "final", final
            return "retry", ForgePilot.retry_notice("model returned an empty <final> answer")
        raw = raw.strip()
        if raw:
            return "final", raw
        return "retry", ForgePilot.retry_notice("model returned an empty response")

    @staticmethod
    def retry_notice(problem=None):
        prefix = "Runtime notice"
        if problem:
            prefix += f": {problem}"
        else:
            prefix += ": model returned malformed tool output"
        return (
            f"{prefix}. Reply with a valid <tool> call or a non-empty <final> answer. "
            'For multi-line files, prefer <tool name="write_file" path="file.py"><content>...</content></tool>.'
        )

    @staticmethod
    def parse_xml_tool(raw):
        match = re.search(r"<tool(?P<attrs>[^>]*)>(?P<body>.*?)</tool>", raw, re.S)
        if not match:
            return None
        attrs = ForgePilot.parse_attrs(match.group("attrs"))
        name = str(attrs.pop("name", "")).strip()
        if not name:
            return None

        body = match.group("body")
        args = dict(attrs)
        for key in ("content", "old_text", "new_text", "command", "task", "pattern", "path"):
            if f"<{key}>" in body:
                args[key] = ForgePilot.extract_raw(body, key)

        body_text = body.strip("\n")
        if name == "write_file" and "content" not in args and body_text:
            args["content"] = body_text
        if name == "delegate" and "task" not in args and body_text:
            args["task"] = body_text.strip()
        return {"name": name, "args": args}

    @staticmethod
    def parse_attrs(text):
        attrs = {}
        for match in re.finditer(r"""([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""", text):
            attrs[match.group(1)] = match.group(2) if match.group(2) is not None else match.group(3)
        return attrs

    @staticmethod
    def extract(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:].strip()
        return text[start:end].strip()

    @staticmethod
    def extract_raw(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:]
        return text[start:end]

    def reset(self):
        self.session["history"] = []
        self.session["memory"].clear()
        self.session["memory"].update(memorylib.default_memory_state())
        self.memory = memorylib.LayeredMemory(self.session["memory"], workspace_root=self.root)
        self.session_store.save(self.session)

    def path(self, raw_path):
        path = Path(raw_path)
        path = path if path.is_absolute() else self.root / path
        resolved = path.resolve()
        # 所有文件类工具都被锚定在 workspace root 之下。
        # 这样既能防住 "../" 逃逸，也能防住符号链接解析后跳出仓库。
        if os.path.commonpath([str(self.root), str(resolved)]) != str(self.root):
            raise ValueError(f"path escapes workspace: {raw_path}")
        return resolved


MiniAgent = ForgePilot
