"""Microduck vertical jump task — attempt 1.

Episodic policy: robot starts standing, crouches, launches into an upright
ballistic hop (all robot-terrain contact broken), lands, and returns to the
HOME stand. Triggered at deployment like kick/roulade (policy switch, zeroed
commands; no phase clock).

Design (jump section of mdp.py has the state machinery):
  • ONE dense task signal — paid increments of the flight-apex frontier
    (jump_height_progress): only height gained during latched upright flight
    counts, only the frontier pays (best jump pays once, re-hopping pays 0).
  • Landing annuity gated on FLIGHT COMPLETION (latch × apex smoothstep
    8–25 mm above stand) — "do nothing" earns nothing, the standing spawn
    cannot farm it, and the gate opens in proportion to real jump height.
  • jump_takeoff_velocity is the discovery bootstrap (upward vz while
    supported, integral-capped at 10 cm per episode — bobbing farms it once).
  • jump_stand_tax (roulade run-3 lesson): post-jump steps below stand height
    are net-negative — "jump then crumple" loses to "stick the landing".
  • fell_over termination KEPT (unlike roulade — falling is failure here),
    so crash landings also forfeit the annuity via episode end.
  • Motion-blockers near zero during discovery (a jump IS a large-|a_z|,
    large-action-rate event); smoothness ramps in late by curriculum
    (standup/roulade timing lesson).

DR / obs / commands mirror the roulade env (61D obs parity, zero-padded
command slots) so the exported ONNX drops into the runtime/simulator stack
unchanged.
"""

import math
from copy import deepcopy

# Symmetry — a vertical hop is left-right symmetric; the mirror loss fights
# one-legged launch asymmetries from day 0.
ENABLE_SYMMETRY = True

# ── Domain randomisation (matched to roulade/standup for sim2real parity) ────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_KD_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = False  # a push mid-flight is incoherent
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

# ── Ranges (matched to the roulade env) ──────────────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003   # ramped to 0.015 via curriculum
HEAD_COM_RANDOMIZATION_RANGE        = 0.003   # ramped to 0.01 via curriculum
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)  # unused (kp DR off)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)    # unused (kd DR off)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# Episode: crouch ~0.5 s + flight ~0.3 s + land/settle ~1.5 s. 3 s leaves a
# window to demonstrate the held stand (the annuity's whole point).
EPISODE_LENGTH_S = 3.0

# Empirically-measured standing trunk height (standup lesson: don't guess).
STAND_Z = 0.115

# Flight-apex completion gate (rise above STAND_Z, metres). 8 mm opens the
# gate a crack, 25 mm opens it fully — XL330s on an 800 g robot are torque-
# poor, so the full-credit bar is deliberately modest; raise after attempt 1
# if the measured apex distribution clears it easily.
JUMP_GATE_LO    = 0.008
JUMP_GATE_HI    = 0.025
JUMP_TARGET_RISE = 0.05   # frontier payout saturates here (metres above stand)

_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_jump_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Microduck vertical-jump environment configuration."""

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # Whole-robot ground contact — the FLIGHT DETECTOR: _update_jump_state
    # reads it (name is load-bearing, _JUMP_SUPPORT_SENSOR); flight = no
    # contact from any robot geom, so a hand/hull/beak touch is not flight.
    robot_ground_cfg = ContactSensorCfg(
        name="robot_ground_contact",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors  = (feet_ground_cfg, self_collision_cfg, robot_ground_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: jump task set ────────────────────────────────────────────────
    # Flight-apex frontier — the one dense task signal. A full TARGET_RISE
    # jump pays ≈ weight × 50 total (roulade_progress accounting).
    cfg.rewards["jump_height_progress"] = RewardTermCfg(
        func=microduck_mdp.jump_height_progress,
        weight=8.0,
        params={"stand_height": STAND_Z, "target_rise": JUMP_TARGET_RISE},
    )

    # Discovery carrot: latched-flight steps, capped at 30 paid steps (0.6 s).
    cfg.rewards["jump_airtime"] = RewardTermCfg(
        func=microduck_mdp.jump_airtime,
        weight=1.0,
        params={"max_paid_steps": 30.0},
    )

    # Discovery bootstrap: upward vz while supported+upright, integral-capped
    # at 10 cm/episode — makes the first liftoff findable, then goes silent.
    cfg.rewards["jump_takeoff"] = RewardTermCfg(
        func=microduck_mdp.jump_takeoff_velocity,
        weight=2.0,
        params={"max_paid_meters": 0.10},
    )

    # Completion-gated standing annuity — the dominant attractor (broad stds:
    # partial landings must score visibly, standup composite lesson).
    cfg.rewards["jump_landing_composite"] = RewardTermCfg(
        func=microduck_mdp.jump_landing_composite,
        weight=4.0,
        params={
            "target_height": STAND_Z,
            "height_std":    0.04,
            "upright_std":   0.40,
            "pose_std":      0.40,
            "joint_indices": _LEG_JOINTS,
            "gate_lo":       JUMP_GATE_LO,
            "gate_hi":       JUMP_GATE_HI,
        },
    )

    # Broad bootstrap layer: linear upright × gate (gradient far from goal).
    cfg.rewards["jump_upright_after"] = RewardTermCfg(
        func=microduck_mdp.jump_upright_after,
        weight=1.5,
        params={
            "target_height": STAND_Z,
            "gate_lo":       JUMP_GATE_LO,
            "gate_hi":       JUMP_GATE_HI,
        },
    )

    # Post-jump crumple tax (SELF-NEGATING → POSITIVE weight, sign convention).
    cfg.rewards["jump_stand_tax"] = RewardTermCfg(
        func=microduck_mdp.jump_stand_tax,
        weight=5.0,
        params={
            "target_height": STAND_Z,
            "gate_lo":       JUMP_GATE_LO,
            "gate_hi":       JUMP_GATE_HI,
        },
    )

    # ── Sim2real regularisers ─────────────────────────────────────────────────
    # Motion-blockers ≈0 during discovery (the jump IS an impact + action-rate
    # event); smoothness ramps in by curriculum after the skill exists.
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.01   # a clean hop is LOW-ω: mild tax ok
    cfg.rewards["angular_momentum"].weight = -0.001
    cfg.rewards.pop("soft_landing", None)

    # |a_z| impact shaping — tiny from step 0 (landing is an impact event;
    # taxing it hard during discovery kills attempts), ramped by curriculum.
    # SELF-NEGATING (returns -|a_z|) → POSITIVE weight.
    cfg.rewards["gentle_landing"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.001,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Sideways-drift guards: a vertical hop should go UP — reuse the roulade
    # flatness/lateral machinery as mild dense pressure (positive quantities,
    # negative weights).
    cfg.rewards["jump_flatness"] = RewardTermCfg(
        func=microduck_mdp.roulade_flatness_penalty,
        weight=-0.3,
    )
    cfg.rewards["jump_lateral_vel"] = RewardTermCfg(
        func=microduck_mdp.roulade_lateral_velocity_penalty,
        weight=-0.3,
    )

    # Self-collision — LIGHT: a deep crouch needs body-on-body proximity.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Always-on upright is FINE here (unlike roulade — nothing to flip), but
    # the gated landing terms already own uprightness; drop the base term to
    # keep the reward mass accounting simple.
    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    # ── Observations (identical layout to walking / roulade policies) ─────────
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # Command obs slots: zero padding for head (4) and body (6) — 61D parity.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # ── Command: tiny noise around zero (kept for obs-shape parity) ──────────
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # ── Terminations ──────────────────────────────────────────────────────────
    # fell_over KEPT (falling is failure here, unlike roulade) + NaN guard.
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    # Explicit STANDING spawn (run-2 fix): the base reset leaves the root at
    # the entity default (z≈0 → collapsed heap), so run 1 learned to jump
    # from a floor pose and holds still when deployed from a clean stand.
    # Reuse the roulade standing bucket: z 0.11–0.12, ±5° tilt noise, HOME
    # joints — exactly the state the browser hands the policy.
    cfg.events["set_jump_spawn"] = EventTermCfg(
        func=microduck_mdp.reset_roulade_state,
        mode="reset",
        params={
            "standing_prob":     1.0,
            "midroll_prob":      0.0,
            "standing_z_min":    0.11,
            "standing_z_max":    0.12,
            "standing_tilt_max": math.radians(5.0),
            "forward_vel_range": (0.0, 0.0),
        },
    )
    # Zero the jump frontier/latch accounting on every reset (runs after the
    # spawn event — dict insertion order).
    cfg.events["reset_jump_state"] = EventTermCfg(
        func=microduck_mdp.reset_jump_state,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    if "push_robot" in cfg.events and not ENABLE_VELOCITY_PUSHES:
        del cfg.events["push_robot"]

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    # ── Terrain ───────────────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curriculum ────────────────────────────────────────────────────────────
    if "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    # Smoothness ramps — introduced after the hop exists (timing lesson:
    # attempt-taxes active during discovery make "do nothing" the argmax).
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.1},
                {"step": 1500 * 24,  "weight": -0.2},
                {"step": 3000 * 24,  "weight": -0.4},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2000 * 24,  "weight": -5e-4},
                {"step": 3000 * 24,  "weight": -1e-3},
            ],
        },
    )
    cfg.curriculum["gentle_landing_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            # POSITIVE weights: the func is self-negating (returns -|a_z|).
            "reward_name":   "gentle_landing",
            "weight_stages": [
                {"step": 0,          "weight": 0.001},
                {"step": 2000 * 24,  "weight": 0.003},
            ],
        },
    )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckJumpRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_jump",
    run_name="microduck_jump",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=10_000,
)
