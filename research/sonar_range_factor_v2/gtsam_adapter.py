"""Optional research-only two-pose CustomFactor, never loaded by the navigator."""
import numpy as np

from .models import profiled_residual_jacobian


def make_m1_factor(track, key_a, key_b):
    """Use only away from profile ties/switches; callback fails closed there."""
    import gtsam

    if len(track.ranges) != 2 or key_a == key_b:
        raise ValueError("Expected two different Pose2 keys and two ranges")

    def callback(_factor, values, jacobians):
        poses = [values.atPose2(k) for k in (key_a, key_b)]
        world = np.asarray([[p.x(), p.y(), p.theta()] for p in poses])
        residual, jac, _ = profiled_residual_jacobian(track, world)
        if jacobians is not None:
            for i, p in enumerate(poses):
                c, s = np.cos(p.theta()), np.sin(p.theta())
                chart = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
                jacobians[i] = np.asfortranarray(jac[:, 3*i:3*i+3]@chart)
        return residual

    return gtsam.CustomFactor(gtsam.noiseModel.Unit.Create(2), [key_a, key_b], callback)
