"""Arm 'IK direction' inputs (--ik): which way each joint should move to close the hand-to-target gap.

Why: the model drives the arm well until the last centimetre, then hovers and drifts out of the 1 cm zone
(the hold test: expert base + our arm holds in only 18/50 scenes; our base + expert arm holds in 50/50).
The torque that corrects a 1 cm error is tiny next to the torques of driving and reaching, so it barely
registers in the training loss. The expert corrects it with geometry: it converts the hand error into
joint speeds through the arm's Jacobian (damped least squares) and lets a joint reference creep along them.

These inputs give the model that same geometric signal, computed from measurements only (joint angles,
base pose, target), the same way the hand position is: forward kinematics + Jacobian of the robot model.
    qd_ik       (7)   the joint speeds the template's IK formula asks for right now (rad/s)
    qd_ik_avg   (14)  their running averages over ~0.5 s and ~2 s: a stand-in for the creeping reference
Nothing comes from the expert's internal state; the formula and its constants are the template's
(arm_controller: cartesian_kp, max_cartesian_speed, reach limit, singularity damping, max joint speed),
without the joint-limit saturation loop.
"""
from _paths import ROOT, find_result  # noqa: F401  (project folder; lets the code folders import each other)
import numpy as np

IK_TAUS_S = (0.5, 2.0)
IK_NAMES = ([f"qd_ik_{i}" for i in range(1, 8)]
            + [f"qd_ik_avg{int(t * 1000)}ms_{i}" for t in IK_TAUS_S for i in range(1, 8)])

_MODEL = None


def ik_rates(dyn_model, ee_frame_id, shoulder_frame_id, q_pin, ee, target, params):
    """Template IK joint speeds. Forward kinematics must already be up to date for q_pin."""
    import pinocchio as pin
    pin.computeJointJacobians(dyn_model.pin_model, dyn_model.pin_data, q_pin)
    J = pin.computeFrameJacobian(dyn_model.pin_model, dyn_model.pin_data, q_pin, ee_frame_id,
                                 pin.LOCAL_WORLD_ALIGNED)[:3, 6:13]
    sigma_min = np.linalg.svd(J, compute_uv=False)[-1]
    damping = np.interp(sigma_min, [0.0, params.singularity_threshold], [params.max_ik_damping, params.min_ik_damping])
    shoulder = dyn_model.pin_data.oMf[shoulder_frame_id].translation
    offset = np.asarray(target, float) - shoulder
    if np.linalg.norm(offset) > params.reach_limit:
        offset = offset * params.reach_limit / np.linalg.norm(offset)
    v = params.cartesian_kp * (shoulder + offset - ee)
    speed = np.linalg.norm(v)
    if speed > params.max_cartesian_speed:
        v = v * params.max_cartesian_speed / speed
    qd = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(3), v)
    m = np.abs(qd).max()
    if m > params.max_joint_vel:
        qd = qd * params.max_joint_vel / m
    return qd


class IKAverager:
    """Running averages of qd_ik (update every 10 ms, like the recordings)."""

    def __init__(self, dt=0.01):
        self.alpha = [np.exp(-dt / t) for t in IK_TAUS_S]
        self.avg = None

    def update(self, qd):
        qd = np.asarray(qd, float)
        if self.avg is None:
            self.avg = [qd.copy() for _ in IK_TAUS_S]
        else:
            self.avg = [a * m + (1 - a) * qd for a, m in zip(self.alpha, self.avg)]
        return np.concatenate([qd] + self.avg).astype(np.float32)


def _model():
    """The robot model (Pinocchio), built once per process from a headless simulator."""
    global _MODEL
    if _MODEL is None:
        import os
        from expert_episode import ARM_REACH_LIMIT, CONF_FILE_NAME, PinWrapper, pb
        from quiet import quiet_native
        from robotnik_arm_control import ArmControllerParams
        here = ROOT
        with quiet_native():
            sim = pb.SimInterface(CONF_FILE_NAME, conf_file_path_ext=here, use_gui=False)
            ext = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
            dm = PinWrapper(CONF_FILE_NAME, "pybullet", ext, ["pybullet"], False, 0, here)
        sim.pybullet_client.disconnect()
        _MODEL = (dm, dm.pin_model.getFrameId("robot_j2s7s300_end_effector"),
                  dm.pin_model.getFrameId("robot_j2s7s300_link_2"), ArmControllerParams(reach_limit=ARM_REACH_LIMIT))
    return _MODEL


def run_ik(arrays):
    """IK inputs for a whole recorded run (rows 10 ms apart), from the recorded measurements.

    Cached in data/.ik_cache (keyed by the run's contents), because the Jacobians take ~0.2 ms per row."""
    import hashlib
    import os
    import pinocchio as pin
    here = ROOT
    key = hashlib.sha1(np.ascontiguousarray(arrays["base_pos"]).tobytes() + np.ascontiguousarray(arrays["q_mes"]).tobytes()
                       + np.ascontiguousarray(arrays["target"][:1]).tobytes()).hexdigest()
    cache = os.path.join(here, "data", ".ik_cache", key + ".npy")
    if os.path.exists(cache):
        return np.load(cache)
    dm, ee_id, sh_id, params = _model()
    avg = IKAverager(0.01)
    out = np.zeros((len(arrays["t"]), len(IK_NAMES)), np.float32)
    for i in range(len(out)):
        q_pin = dm.ReoderJoints2PinVec(np.concatenate((arrays["base_pos"][i], arrays["base_quat"][i],
                                                       arrays["q_mes"][i])), "pos")
        pin.forwardKinematics(dm.pin_model, dm.pin_data, q_pin)
        pin.updateFramePlacements(dm.pin_model, dm.pin_data)
        ee = dm.pin_data.oMf[ee_id].translation.copy()
        out[i] = avg.update(ik_rates(dm, ee_id, sh_id, q_pin, ee, arrays["target"][i], params))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache + ".tmp.npy", out)
    os.replace(cache + ".tmp.npy", cache)
    return out
