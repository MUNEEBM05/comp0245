"""A memory-free ("Markov") version of the expert, used only to LABEL states for DAgger.

Why: DAgger asks the expert "what would you do here?" in states our model drives into. The template
expert answers from its memory (a phase that only moves forward: align -> drive -> orient -> done, an
arm reference that creeps, and a 25 s navigation timeout). The same situation can then get different
answers depending on that memory, which no model can learn from the measurements (the theory calls
this a realisability failure). Classic DAgger with the template expert made our model much worse.

This labeller uses the template's controllers and gains UNCHANGED (base_controller, arm_controller);
only the bookkeeping is replaced: the phase is worked out from what can be measured right now, so
the same measured situation always gets the same answer.

    arm already out (reaching has started)  -> keep reaching, unless the base is clearly off its spot
                                              (> 0.25 m, or facing > 0.2 rad off): then tuck and re-park
    otherwise, far from the parking spot     -> "drive" if the robot has been moving forward over the
                                              last 0.1 s and roughly faces the spot, else "align"
                                              (turn on the spot); "orient" once past the spot
    at the parking spot (< 8 cm, or stopped
    within 15 cm, or just driven past it)    -> "orient" (turn to face the target) until within 0.05 rad,
                                              then "done" (reach)
The arm reference restarts from the measured arm angles whenever reaching starts (the template's
reference equals the tucked pose there, so on the expert's own drives this changes nothing).
The navigation timeout is ignored (it is a clock, not something the robot can see).
check_agreement() measures how often it gives the template expert's answer on the expert's own drives.
"""
from collections import deque

import numpy as np

from robotnik_arm_control import BaseNavState, arm_controller, base_controller

ARM_OUT_RAD = 0.10         # arm counts as "out" once any joint is this far from the tucked pose
REPARK_DIST = 0.25         # m: while reaching, the base is only sent back if it is further than this...
REPARK_FACE = 0.20         # rad: ...or facing further off than this
POS_TOL, ANG_TOL = 0.08, 0.05   # the template's own parking tolerances
MOVING_FWD = 0.003         # m moved forward in the last 0.1 s => "driving" (template drives at >= 0.05 m/s)
NEAR_DIST = 0.15           # m: stopped this close to the spot counts as parked (the base drifts while turning)
DRIVE_MAX_ERR = 1.0        # rad: never call it "driving" when facing further off the spot than this


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def markov_phase(base_pos, yaw, fwd_moved, goal, goal_bearing, q_arm, q_hold):
    """The phase the expert should be in, from measurements only. fwd_moved = forward distance over 0.1 s."""
    to_goal = goal - base_pos[:2]
    dist = float(np.hypot(*to_goal))
    head_err = _wrap(np.arctan2(to_goal[1], to_goal[0]) - yaw)
    face_err = _wrap(goal_bearing - yaw)
    arm_out = float(np.abs(q_arm - q_hold).max()) > ARM_OUT_RAD
    if arm_out and dist < REPARK_DIST and abs(face_err) < REPARK_FACE:
        return "done"
    near = dist < NEAR_DIST and fwd_moved <= MOVING_FWD             # stopped close to the spot
    if dist < POS_TOL or near or (dist < NEAR_DIST and np.cos(head_err) < 0.0):   # at / just past the spot
        return "done" if abs(face_err) < ANG_TOL else "orient"
    if fwd_moved > MOVING_FWD and abs(head_err) < DRIVE_MAX_ERR:
        return "drive"
    return "align"


class MarkovExpert:
    """Template controllers + measured phase. Call reset() per scene, then label() every 1 ms step."""

    def __init__(self, world, target, goal, goal_bearing, base_params, arm_params, q_hold, dt):
        self.w, self.target, self.goal, self.tb = world, np.asarray(target, float), goal, goal_bearing
        self.base_params, self.arm_params, self.q_hold, self.dt = base_params, arm_params, q_hold, dt
        self.q_des, self.reaching = q_hold.copy(), False
        self.lag = max(1, int(round(0.1 / dt)))
        self.past = deque(maxlen=self.lag + 1)

    def label(self, base_pos, yaw, q_mes_arm, qd_mes_arm, q_pin, ee):
        self.past.append(base_pos[:2].copy())
        d = base_pos[:2] - self.past[0]
        fwd = float(np.cos(yaw) * d[0] + np.sin(yaw) * d[1])
        phase = markov_phase(base_pos, yaw, fwd, self.goal, self.tb, q_mes_arm, self.q_hold)
        st = BaseNavState(phase="orient" if phase == "done" else phase, done=(phase == "done"))
        wheels = base_controller(st, base_pos, yaw, self.goal, self.tb, 0.0, self.base_params)  # t=0: no timeout
        reach = phase == "done"
        if reach and not self.reaching:
            self.q_des = q_mes_arm.copy()          # reaching starts from where the arm actually is
        self.reaching = reach
        torque, self.q_des, qd_des = arm_controller(
            self.w.dyn_model, q_pin, q_mes_arm, qd_mes_arm, self.q_des, self.q_hold, self.target, ee,
            self.w.ee_frame_id, self.w.shoulder_frame_id, self.w.arm_lower, self.w.arm_upper, self.dt,
            self.arm_params, reach_active=reach)
        return wheels, torque, self.q_des.copy(), qd_des, phase


def check_agreement(run_dirs=("data/runs",), limit=None):
    """Offline check on the expert's own recordings: does the measured phase match the template's?

    Uses the logged 100 Hz rows (forward motion over 10 rows = 0.1 s). Returns overall agreement, per
    template phase, and how much the wheel commands would differ (rad/s).
    """
    import glob
    import os
    from check_runs import load_run
    from robotnik_arm_control import BaseControllerParams
    from expert_episode import BASE_NAV_TIMEOUT
    names = ["align", "drive", "orient", "done"]
    bp_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)
    agree, total, by_phase, wheel_gap = 0, 0, {n: [0, 0] for n in names}, []
    files = [f for d in run_dirs for f in sorted(glob.glob(os.path.join(d, "run_*.npz")))][:limit]
    for f in files:
        meta, a = load_run(f)
        goal, tb = np.array(meta["base_goal_pos"]), meta["base_goal_bearing"]
        q_hold = a["q_mes"][0, 4:11]
        for i in range(len(a["t"])):
            j = max(0, i - 10)
            d = a["base_pos"][i, :2] - a["base_pos"][j, :2]
            yaw = float(a["base_bearing"][i])
            fwd = np.cos(yaw) * d[0] + np.sin(yaw) * d[1]
            ph = markov_phase(a["base_pos"][i], yaw, fwd, goal, tb, a["q_mes"][i, 4:11], q_hold)
            true = names[int(a["phase"][i])]
            ok = ph == true
            agree += ok
            total += 1
            by_phase[true][0] += ok
            by_phase[true][1] += 1
            st = BaseNavState(phase="orient" if ph == "done" else ph, done=(ph == "done"))
            w = base_controller(st, a["base_pos"][i], yaw, goal, tb, 0.0, bp_params)
            wheel_gap.append(np.abs(w - a["wheel_cmds"][i]).mean())
    wheel_gap = np.array(wheel_gap)
    return {"rows": total, "phase_agreement": agree / total,
            "per_phase": {k: (v[0] / v[1] if v[1] else None, v[1]) for k, v in by_phase.items()},
            "wheel_gap_mean": float(wheel_gap.mean()), "wheel_rows_within_0.1": float((wheel_gap < 0.1).mean())}


if __name__ == "__main__":
    import json
    print(json.dumps(check_agreement(), indent=1))


class MarkovExpertPolicy:
    """The memory-free expert driving the robot itself (sanity check: it must still be a good driver).

    Use as the policy 'markov_expert' in evaluate.py.
    """
    name = "markov_expert"

    def reset(self, world, target):
        from expert_episode import ARM, ARM_REACH_LIMIT, BASE_NAV_TIMEOUT, STANDOFF_POINTS, STANDOFF_Z_POINTS
        from robotnik_arm_control import ArmControllerParams, BaseControllerParams
        target = np.asarray(target, float)
        standoff = float(np.interp(target[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
        tb = float(np.arctan2(target[1], target[0])) if np.linalg.norm(target[:2]) > 1e-4 else 0.0
        goal = target[:2] - standoff * np.array([np.cos(tb), np.sin(tb)])
        self.arm = ARM
        self.x = MarkovExpert(world, target, goal, tb, BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT),
                              ArmControllerParams(reach_limit=ARM_REACH_LIMIT),
                              world.init_joint_angles[ARM].copy(), world.dt)

    def act(self, obs):
        w, tq, _, _, _ = self.x.label(obs["base_pos"], obs["base_bearing"], obs["q_mes"][self.arm],
                                      obs["qd_mes"][self.arm], obs["q_pin"], obs["ee_pos"])
        return w, tq
