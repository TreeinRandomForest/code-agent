from datasets import load_dataset

from typing import Optional
from pathlib import Path
import asyncio
import subprocess
import tempfile
import os, re

from verifiers.base import PatchContext
from verifiers.ast_check import ASTCheckVerifier

#eventually will create ABC class for datasets with interface below
ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

sympy_tasks = [x for x in ds if x["repo"] == "sympy/sympy"]
print(f'# sympy tasks = {len(sympy_tasks)})')

def setup_task(task, 
               install_loc,
               include_hints=False,
               ):

    repo = f"https://github.com/{task['repo']}.git"
    base_commit = task['base_commit']
    env_commit = task['environment_setup_commit']

    prompt = task['problem_statement']
    if include_hints:
        prompt += task['hints_text']

    if not os.path.exists(install_loc):
        os.makedirs(install_loc)

    print(f"Cloning repo at {repo}")
    run(['git', 'clone', repo, install_loc])
    run(['git', 'checkout', env_commit], cwd=install_loc)
    run(['pip', 'install', '-e', '.'], cwd=install_loc)
    run(['git', 'checkout', base_commit], cwd=install_loc)
    
    return prompt


def apply_patch(install_loc: str, patch_text: str):
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.diff', delete=False) as f:
            f.write(patch_text)
            patch_file = f.name

        run(['git', 'apply', patch_file], cwd=install_loc)
    finally:
        os.unlink(patch_file)

def reset_repo(install_loc: str):
    run(['git', 'checkout', '.'], cwd=install_loc)
    run(['git', 'clean', '-fd'], cwd=install_loc)

def run(cmd: list[str], cwd: Optional[str] = None, error_msg: Optional[str] = None):
    ret = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if ret.returncode != 0:
        msg = error_msg or f"Command Failure: {' '.join(cmd)}"
        raise ValueError(msg)
    return ret

def get_changed_files(patch_text):
    return re.findall(r'^diff --git a/(.+?) b/', patch_text, re.MULTILINE)

def create_patch_context(task, patch_text, install_loc):
    return PatchContext(repo_path=Path(install_loc),
                        patch_diff=patch_text,
                        changed_files=get_changed_files(patch_text),
                        task_id=task['instance_id'],
                        ground_truth_patch=task['patch'],
                        )

async def run_verifiers(task, patch_text, install_loc):
    ctx = create_patch_context(task, patch_text, install_loc)
    res = await ASTCheckVerifier().safe_verify(ctx)
    return res
    