"""Opt-in equivalent patch mean/std reductions for the frozen feature builder.

Keep OpenCV patch resampling, per-row norm, corner/return ownership order and
all feature decisions unchanged. This module never installs itself on import.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import textwrap

import numpy as np

EXPECTED_CANONICAL_SHA256 = "31f951d2537acbda871e47ad891b7c6573a46b1ed9d7c3cba85a198bd6a39486"


def _normalized_patches(cv2, padded_image, rounded_columns, rounded_rows, usable, radius):
    if not len(usable):
        return np.empty((0, 64), dtype=np.float32), np.empty(0, dtype=np.float32)
    patches = []
    for index in usable:
        row, column = int(rounded_rows[index]), int(rounded_columns[index])
        patch = padded_image[row:row+2*radius+1, column:column+2*radius+1]
        patches.append(cv2.resize(patch, (8, 8), interpolation=cv2.INTER_AREA).reshape(-1))
    descriptors = np.stack(patches)
    descriptors -= np.mean(descriptors, axis=1)[:, None]
    deviations = np.std(descriptors, axis=1)
    descriptors /= np.where(deviations > 1e-6, deviations, 1.)[:, None]
    return descriptors, deviations


def build_batched_canonical(original):
    source = textwrap.dedent(inspect.getsource(original))
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != EXPECTED_CANONICAL_SHA256:
        raise ValueError("unrecognized feature builder; refusing deployment transformation")
    tree = ast.parse(source)
    method = tree.body[0]
    loops = [n for n in method.body if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == "corner_index"]
    if len(loops) != 1:
        raise ValueError("ambiguous descriptor loop")
    loop = loops[0]
    patch = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == "patch")
    norm = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.AugAssign)
                and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "max")
    guard = next(copy.deepcopy(n) for n in loop.body[patch:norm] if isinstance(n, ast.If))
    loop.body[patch:norm] = ast.parse('''
descriptor = hil_descriptors[descriptor_index]
deviation = float(hil_deviations[descriptor_index])
''').body + [guard]
    loop.target = ast.parse("descriptor_index, corner_index = 0, 0").body[0].targets[0]
    loop.iter = ast.parse("enumerate(usable)", mode="eval").body
    offset = method.body.index(loop)
    method.body[offset:offset] = ast.parse('''
hil_descriptors, hil_deviations = _hil_normalized_patches(cv2, padded_image,
    rounded_columns, rounded_image_rows, usable, patch_radius)
''').body
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__, _hil_normalized_patches=_normalized_patches)
    exec(compile(tree, "<hil-batched-feature-reductions>", "exec"), namespace)
    candidate = namespace[original.__name__]
    candidate.frozen_method_sha256 = digest
    return candidate
