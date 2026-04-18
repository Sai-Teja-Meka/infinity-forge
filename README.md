# ∞ Forge

∞ Forge is a self-compounding verified Python function library. A generator proposes candidate Python functions; a cascade of static, dynamic, and semantic checks accepts or rejects each one. Accepted functions become building blocks that the generator can reference when producing the next round of candidates, so the library's expressive power compounds over time. The long-term goal is an unattended, overnight-safe process that continuously grows a corpus of verified, composable utilities.

The system is only as trustworthy as its weakest link, and the weakest link is the execution environment that runs unverified candidate code. If the sandbox leaks — by exhausting memory, writing to disk, opening a network socket, or simply failing to kill a runaway subprocess — the whole forge becomes unsafe to run autonomously. So the sandbox is built first, tested exhaustively against a battery of adversarial inputs, and only then do later layers get wired in on top of it.

## Day 1: sandbox only

This repository currently contains only the sandbox and its adversarial test battery. There is no generator, no cascade, no storage, and no novelty layer. The goal of Day 1 is a single public function — `infinity_forge.sandbox.run_in_sandbox` — that executes a candidate Python function in an isolated subprocess with CPU, memory, file-descriptor, process, and file-size limits, enforces a wall-clock timeout, and returns a structured result. Later days will build the cascade on top of this foundation.
