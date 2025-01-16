"""Field of view computation."""

import numpy as np


class FieldOfView(object):
    """Compute field of view prefactors.

    The field of view angle twophi is given in degrees.
    out_of_view_factor is C in the paper.
    """

    def __init__(self, phi=None, out_of_view_factor=None):
        phi = phi
        out_of_view_factor = out_of_view_factor
        cosphi = np.cos(phi / 180.0 * np.pi)
        self.cosphi = np.expand_dims(cosphi, axis=0)
        self.out_of_view_factor = np.expand_dims(out_of_view_factor, axis=0)

    def __call__(self, desired_direction, forces_direction):
        """Weighting factor for field of view.

        desired_direction : e, rank 2 and normalized in the last index.
        forces_direction : f, rank 3 tensor.
        """
        in_sight = (
            np.einsum("aj,abj->ab", desired_direction, forces_direction)
            > np.linalg.norm(forces_direction, axis=-1) * self.cosphi
        )
        out = self.out_of_view_factor * np.ones_like(in_sight)
        out[in_sight] = 1.0
        np.fill_diagonal(out, 0.0)
        return out
