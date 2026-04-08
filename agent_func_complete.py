# agent_func.py
# OpenRLHF loads this file via --agent_func_path and looks for AgentExecutor.
#
# Flow:
# 1. OpenRLHF creates an AgentInstance per episode
# 2. Calls reset() with states from the JSONL dataset
# 3. Feeds observation to vLLM, which generates action tokens
# 4. Calls step() with the generated action text
# 5. Appends environment_feedback to conversation
# 6. Repeats 3-5 until done=True
# 7. Uses final rewards for GRPO advantage computation

import os
import re
import json
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict

import torch
from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor

from verifiers.ast_check import ASTCheckVerifier
from verifiers.base import PatchContext, VerifierResult, VerifierStatus

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------Configuration-----

CHAT_TEMPLATE = os.environ.get("CHAT_TEMPLATE", "qwen")


# ------Utilities-----

def wrap_feedback(text: str, done: bool) -> str:
    """Wrap observation text in the model's chat template."""
    if CHAT_TEMPLATE == "qwen":
        if not done:
            return (
                f"<|im_start|>user\n{text}\n"
                f"Use the available tools to continue.<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            return f"<|im_start|>user\n{text}<|im_end|>"
    else:
        return text


def parse_tool_calls(action_text: str) -> list[dict[str, Any]]:
    """
    Parse tool calls from model output.

    Expected format:
    <tool_call>{"name": "read_file", "arguments": {"path": "foo.py"}}</tool_call>
    """
    calls = []
    for match in re.finditer(r'<tool_call>(.*?)</tool_call>', action_text, re.DOTALL):
        try:
            call = json.loads(match.group(1))
            if "name" in call:
                calls.append({
                    "name": call["name"],
                    "arguments": call.get("arguments", {}),
                })
        except json.JSONDecodeError:
            continue
    return calls


def truncate(text: str, max_chars: int = 8000) -> str:
    """Truncate long output, keeping beginning and end."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n... truncated ...\n\n" + text[-half:]


# -----Agent Instance (Environment Definition)--------------

class AgentInstance(AgentInstanceBase):

    async def __init__(self, *args, **kwargs):
        self.step_idx = 0
        self.max_steps = 20

        # Cheap verifiers run after every file edit (dense feedback)
        self.dense_verifiers = [ASTCheckVerifier(timeout=10.0)]

        # Comprehensive verifiers run on final patch submission
        # For minimal loop, same as dense. Add LinterVerifier, UnitTestVerifier etc. later.
        self.final_verifiers = [ASTCheckVerifier(timeout=60.0)]

        self.repo_path = None
        self.task_metadata = None

    async def reset(self, states: dict, **kwargs):
        """
        Called at episode start.

        states["observation"]: prompt text from JSONL dataset
        states["label"]: JSON string with task metadata (repo_path, instance_id)

        For GRPO (multiple samples per task), OpenRLHF calls reset()
        for each sample, so we reset the repo to clean state here.
        """
        self.step_idx = 0
        self.task_metadata = json.loads(states.get("label", "{}"))
        self.repo_path = Path(self.task_metadata.get("repo_path", "/tmp/swe-bench-repo"))

        await asyncio.to_thread(self._reset_repo)

        return {"observation": states["observation"]}

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        """
        Called after each model generation (of patch).

        Flow:
        1. Parse tool calls from action_text
        2. Execute tools on the repo (inline)
        3. If files were modified, run dense verifiers
        4. If patch submitted, run final verifiers
        5. Format observations and return
        """
        action_text = states["action_text"]
        self.step_idx += 1

        logger.info(f"Step {self.step_idx}/{self.max_steps}")

        #-----Parse tool calls-----------

        tool_calls = parse_tool_calls(action_text)

        if not tool_calls:
            return {
                "rewards": torch.tensor(0.0),
                "scores": torch.tensor(0.0),
                "environment_feedback": wrap_feedback(
                    "No tool call detected. Use format: "
                    '<tool_call>{"name": "...", "arguments": {...}}</tool_call>',
                    done=False,
                ),
                "done": self.step_idx >= self.max_steps,
            }

        #------Execute tool calls (inline)-------------

        observations = []
        submitted_patch = None
        files_modified = False

        for tc in tool_calls:
            name = tc["name"]
            args = tc["arguments"]

            try:
                if name == "read_file":
                    path = self.repo_path / args["path"]
                    if not path.exists():
                        observations.append(f"[read_file]: File not found: {args['path']}")
                        continue
                    content = path.read_text(encoding="utf-8", errors="replace")
                    lines = content.splitlines()
                    start = args.get("start_line", 1) - 1
                    end = args.get("end_line", len(lines))
                    numbered = [
                        f"{i + start + 1:6d}\t{l}"
                        for i, l in enumerate(lines[start:end])
                    ]
                    output = truncate("\n".join(numbered))
                    observations.append(f"[read_file]: {args['path']}\n{output}")

                elif name == "edit_file":
                    path = self.repo_path / args["path"]
                    if not path.exists():
                        observations.append(f"[edit_file]: File not found: {args['path']}")
                        continue
                    content = path.read_text(encoding="utf-8", errors="replace")
                    old_str = args["old_str"]
                    count = content.count(old_str)
                    if count == 0:
                        observations.append(f"[edit_file]: old_str not found in {args['path']}")
                    elif count > 1:
                        observations.append(
                            f"[edit_file]: old_str found {count} times, must be unique"
                        )
                    else:
                        new_content = content.replace(old_str, args["new_str"], 1)
                        path.write_text(new_content, encoding="utf-8")
                        files_modified = True
                        observations.append(f"[edit_file]: Replaced 1 occurrence in {args['path']}")

                elif name == "search":
                    proc = await asyncio.create_subprocess_exec(
                        "grep", "-rn", "--include=*.py",
                        "-m", str(args.get("max_results", 20)),
                        args["pattern"],
                        args.get("path", "."),
                        cwd=str(self.repo_path),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
                    output = stdout.decode("utf-8", errors="replace")
                    if not output.strip():
                        output = "No matches found"
                    observations.append(f"[search]: {truncate(output)}")

                elif name == "run_command":
                    proc = await asyncio.create_subprocess_shell(
                        args["command"],
                        cwd=str(self.repo_path),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=60
                    )
                    output = (
                        stdout.decode("utf-8", errors="replace")
                        + stderr.decode("utf-8", errors="replace")
                    )
                    observations.append(f"[run_command]: {truncate(output)}")

                elif name == "list_dir":
                    target = args.get("path", ".")
                    depth = str(args.get("depth", 2))
                    proc = await asyncio.create_subprocess_exec(
                        "find", target,
                        "-maxdepth", depth,
                        "-not", "-path", "*/.git/*",
                        "-not", "-path", "*/__pycache__/*",
                        cwd=str(self.repo_path),
                        stdout=asyncio.subprocess.PIPE,
                    )
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                    observations.append(f"[list_dir]:\n{truncate(stdout.decode())}")

                elif name == "submit_patch":
                    proc = await asyncio.create_subprocess_exec(
                        "git", "diff", "HEAD",
                        cwd=str(self.repo_path),
                        stdout=asyncio.subprocess.PIPE,
                    )
                    stdout, _ = await proc.communicate()
                    diff = stdout.decode("utf-8", errors="replace")
                    if diff.strip():
                        submitted_patch = diff
                        observations.append(
                            f"[submit_patch]: Patch submitted ({len(diff)} chars)"
                        )
                    else:
                        observations.append("[submit_patch]: No changes to submit")

                else:
                    observations.append(f"[error]: Unknown tool '{name}'")

            except asyncio.TimeoutError:
                observations.append(f"[{name}]: Timed out")
            except Exception as e:
                observations.append(f"[{name}]: Error - {type(e).__name__}: {e}")

        #------------Run verifiers--------------

        verifier_feedback = ""

        if submitted_patch:
            # Final submission: run comprehensive verifiers
            results = await self._run_verifier_set(self.final_verifiers)
            verifier_feedback = self._format_verifier_results(results, prefix="FINAL")
            reward = self._compute_reward(results)

        elif files_modified:
            # Intermediate edit: run cheap verifiers for dense feedback
            results = await self._run_verifier_set(self.dense_verifiers)
            verifier_feedback = self._format_verifier_results(results, prefix="CHECK")
            reward = 0.0  # no intermediate reward, just feedback

        else:
            reward = 0.0

        # -----------Determine termination----------------

        if submitted_patch:
            done = True
        elif self.step_idx >= self.max_steps:
            done = True
            reward = -1.0
            verifier_feedback += "\n[TIMEOUT] Maximum steps reached without submission."
        else:
            done = False

        feedback_text = "\n\n".join(observations)
        if verifier_feedback:
            feedback_text += "\n" + verifier_feedback

        logger.info(
            f"Step {self.step_idx}: done={done}, reward={reward:.3f}, "
            f"tools={[tc['name'] for tc in tool_calls]}"
        )

        return {
            "rewards": torch.tensor(reward),
            "scores": torch.tensor(max(0.0, reward)),
            "environment_feedback": wrap_feedback(feedback_text, done=done),
            "done": done,
            "extra_logs": {
                "step_count": torch.tensor(self.step_idx),
            },
        }

    # --------Helpers-----------------

    def _reset_repo(self):
        """Reset repo to clean state. Runs synchronously via asyncio.to_thread."""
        import subprocess
        subprocess.run(
            ["git", "checkout", "."],
            cwd=str(self.repo_path),
            capture_output=True,
        )
        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=str(self.repo_path),
            capture_output=True,
        )

    async def _get_changed_files(self) -> list[str]:
        """Get list of Python files modified relative to HEAD."""
        proc = await asyncio.create_subprocess_exec(
            "git", "diff", "--name-only", "HEAD",
            cwd=str(self.repo_path),
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return [
            f.strip() for f in stdout.decode().splitlines()
            if f.strip().endswith(".py")
        ]

    async def _run_verifier_set(self, verifiers: list) -> list[VerifierResult]:
        """Run a set of verifiers in parallel on currently changed files."""
        changed = await self._get_changed_files()

        if not changed:
            return [
                VerifierResult(
                    name="no_changes",
                    status=VerifierStatus.PASS,
                    score=1.0,
                    details={"message": "No Python files changed"},
                )
            ]

        ctx = PatchContext(
            repo_path=self.repo_path,
            patch_diff="",
            changed_files=changed,
            task_id=self.task_metadata.get("instance_id", "unknown"),
        )
        results = await asyncio.gather(*[v.safe_verify(ctx) for v in verifiers])
        return list(results)

    def _format_verifier_results(
        self, results: list[VerifierResult], prefix: str = "CHECK"
    ) -> str:
        """Format verifier results as human-readable feedback."""
        lines = []
        for r in results:
            status_str = "PASS" if r.passed else "FAIL"
            lines.append(f"[{prefix}/{r.name}]: {status_str} (score={r.score:.2f})")
            if not r.passed and r.details.get("errors"):
                for e in r.details["errors"][:5]:  # cap at 5 errors
                    lines.append(f"  {e.get('file', '?')}:{e.get('line', '?')}: {e.get('message', '?')}")
        return "\n".join(lines)

    def _compute_reward(self, results: list[VerifierResult]) -> float:
        """
        Compute scalar reward from final verifier results.

        For minimal loop: average of scores.
        Later: use RewardComposer with hierarchical/weighted modes.
        """
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)


# -----OpenRLHF entry point-------

class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(self):
        super().__init__(AgentInstance)
