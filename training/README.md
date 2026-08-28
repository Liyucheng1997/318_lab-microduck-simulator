# Jump policy training

The `jump.onnx` policy in `app/public/policies/` was trained with
[apirrone/mjlab_microduck](https://github.com/apirrone/mjlab_microduck)
(mjlab / MuJoCo Warp + PPO via rsl_rl), with a new task added by this fork:

- `microduck_jump_env_cfg.py` — the `Mjlab-Jump-Flat-MicroDuck` env config
  (drop into `src/mjlab_microduck/tasks/` and register in `tasks/__init__.py`)
- `mdp_with_jump.py` — the full `tasks/mdp.py` including the jump section
  appended at the end (state accumulator, flight latch, apex-frontier
  progress reward, completion-gated landing annuity, stand tax)

## Task design

Episodic one-shot: standing start → crouch → upright ballistic hop (all
robot-terrain contact broken) → land → return to the HOME stand. 61D obs /
14D action, identical policy interface to every other Microduck policy, so
the runtime and the browser sim hot-swap it like kick/roll.

Anti-reward-hacking structure (per the repo's AGENTS.md lessons):

- **Flight** = no contact AND trunk upright; a **latch** requires 2
  consecutive flight steps (spawn-settle can't open the gates).
- **Apex frontier pays once**: only new per-episode max flight height is
  rewarded — re-hopping to the same height earns zero.
- **Landing annuity** (standing composite × smoothstep on apex 8–25 mm
  above stand height): "never jumped" and "jumped 2 mm" earn ≈nothing.
- **Stand tax** below standing height after the jump: "jump then crumple"
  is net-negative.
- Smoothness/impact penalties ramp in by curriculum only after discovery.

## Reproduce

```bash
git clone https://github.com/apirrone/mjlab_microduck
# copy the two files as described above, register the task, then:
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 2000
uv run scripts/export.py Mjlab-Jump-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/microduck_jump/<run>/model_2000.pt \
  --onnx-file jump.onnx
```

Trained 2026-08-28 on an RTX 5080 Laptop GPU (~85 min, 4096 envs, obs
normalizer baked into the ONNX by export.py).
