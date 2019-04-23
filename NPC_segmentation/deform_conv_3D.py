from __future__ import absolute_import, division

import numpy as np
from scipy.ndimage.interpolation import map_coordinates as sp_map_coordinates
import tensorflow as tf


def tf_flatten(a):
    """Flatten tensor"""
    return tf.reshape(a, [-1])


def tf_repeat(a, repeats, axis=0):
    """TensorFlow version of np.repeat for 1D"""
    # https://github.com/tensorflow/tensorflow/issues/8521
    assert len(a.get_shape()) == 1

    a = tf.expand_dims(a, -1)
    a = tf.tile(a, [1, repeats])
    a = tf_flatten(a)
    return a


def tf_repeat_3d(a, repeats):
    """Tensorflow version of np.repeat for 3D"""

    assert len(a.get_shape()) == 2
    a = tf.expand_dims(a, 0)
    a = tf.tile(a, [repeats, 1, 1])
    return a


def tf_map_coordinates(input, coords, order=1):
    """Tensorflow verion of scipy.ndimage.map_coordinates

    Note that coords is transposed and only 2D is supported

    Parameters
    ----------
    input : tf.Tensor. shape = (s, s)
    coords : tf.Tensor. shape = (n_points, 2)
    """

    assert order == 1

    coords_lt = tf.cast(tf.floor(coords), 'int32')
    coords_rb = tf.cast(tf.ceil(coords), 'int32')
    coords_lb = tf.stack([coords_lt[:, 0], coords_rb[:, 1]], axis=1)
    coords_rt = tf.stack([coords_rb[:, 0], coords_lt[:, 1]], axis=1)

    vals_lt = tf.gather_nd(input, coords_lt)
    vals_rb = tf.gather_nd(input, coords_rb)
    vals_lb = tf.gather_nd(input, coords_lb)
    vals_rt = tf.gather_nd(input, coords_rt)

    coords_offset_lt = coords - tf.cast(coords_lt, 'float32')
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, 1]

    return mapped_vals


def sp_batch_map_coordinates(inputs, coords):
    """Reference implementation for batch_map_coordinates"""
    coords = coords.clip(0, inputs.shape[1] - 1)
    mapped_vals = np.array([
        sp_map_coordinates(input, coord.T, mode='nearest', order=1)
        for input, coord in zip(inputs, coords)
    ])
    return mapped_vals


def tf_batch_map_coordinates(input, coords, order=1):
    """Batch version of tf_map_coordinates

    Only supports 2D feature maps

    Parameters
    ----------
    input : tf.Tensor. shape = (b, s, s, d)
    coords : tf.Tensor. shape = (b, n_points, 2)
    """

    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    input_size = input_shape[1]
    input_d_size = input_shape[3]
    n_coords = tf.shape(coords)[1]

    coords = tf.clip_by_value(coords, 0, tf.cast(input_size, 'float32') - 1)

    coords_ltb = tf.cast(tf.floor(coords), 'int32')
    coords_rbf = tf.cast(tf.ceil(coords), 'int32')

    coords_ltf = tf.stack([coords_ltb[..., 0], coords_rbf[..., 1],coords_ltb[..., 2]], axis=-1)
    coords_rtb = tf.stack([coords_rbf[..., 0], coords_ltb[..., 1], coords_ltb[..., 2]], axis=-1)
    coords_rtf = tf.stack([coords_rbf[..., 0], coords_rbf[..., 1], coords_ltb[..., 2]], axis=-1)

    coords_lbb = tf.stack([coords_ltb[..., 0], coords_ltb[..., 1], coords_rbf[..., 2]], axis=-1)
    coords_lbf = tf.stack([coords_ltb[..., 0], coords_rbf[..., 1], coords_rbf[..., 2]], axis=-1)
    coords_rbb = tf.stack([coords_rbf[..., 0], coords_ltb[..., 1], coords_rbf[..., 2]], axis=-1)

    idx = tf_repeat(tf.range(batch_size), n_coords)

    def _get_vals_by_coords(input, coords):
        indices = tf.stack([
            idx, tf_flatten(coords[..., 0]), tf_flatten(coords[..., 1]), tf_flatten(coords[..., 2])
        ], axis=-1)
        vals = tf.gather_nd(input, indices)
        vals = tf.reshape(vals, (batch_size, n_coords))
        return vals

    vals_ltb = _get_vals_by_coords(input, coords_ltb)
    vals_rbf = _get_vals_by_coords(input, coords_rbf)
    vals_ltf = _get_vals_by_coords(input, coords_ltf)
    vals_rtb = _get_vals_by_coords(input, coords_rtb)
    vals_rtf = _get_vals_by_coords(input, coords_rtf)
    vals_lbb = _get_vals_by_coords(input, coords_lbb)
    vals_lbf = _get_vals_by_coords(input, coords_lbf)
    vals_rbb = _get_vals_by_coords(input, coords_rbb)

    coords_offset_ltb = coords - tf.cast(coords_ltb, 'float32')
    coords_offset_lbb = coords - tf.cast(coords_lbb, 'float32')

    vals_tb = vals_ltb + (vals_rtb - vals_ltb) * coords_offset_ltb[..., 0]
    vals_tf = vals_ltf + (vals_rtf - vals_ltf) * coords_offset_ltb[..., 0]
    mapped_vals_t = vals_tb + (vals_tf - vals_tb) * coords_offset_ltb[..., 1]
    vals_bb = vals_lbb + (vals_rbb - vals_lbb) * coords_offset_lbb[..., 0]
    vals_bf = vals_lbf + (vals_rbf - vals_lbf) * coords_offset_lbb[..., 0]
    mapped_vals_b = vals_bb + (vals_bf - vals_bb) * coords_offset_lbb[..., 1]
    mapped_vals = mapped_vals_t + (mapped_vals_b - mapped_vals_t) * coords_offset_ltb[..., 2]

    return mapped_vals


def sp_batch_map_offsets(input, offsets):
    """Reference implementation for tf_batch_map_offsets"""

    batch_size = input.shape[0]
    input_size = input.shape[1]

    offsets = offsets.reshape(batch_size, -1, 2)
    grid = np.stack(np.mgrid[:input_size, :input_size], -1).reshape(-1, 2)
    grid = np.repeat([grid], batch_size, axis=0)
    coords = offsets + grid
    coords = coords.clip(0, input_size - 1)

    mapped_vals = sp_batch_map_coordinates(input, coords)
    return mapped_vals


def tf_batch_map_offsets(input, offsets, order=1):
    """Batch map offsets into input

    Parameters
    ---------
    input : tf.Tensor. shape = (b, s, s, d)
    offsets: tf.Tensor. shape = (b, s, s, d, 3)
    """

    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    input_size = input_shape[1]
    input_d_size = input_shape[3]

    offsets = tf.reshape(offsets, (batch_size, -1, 3))
    grid = tf.meshgrid(
        tf.range(input_size), tf.range(input_size), tf.range(input_d_size),indexing='ij'
    )
    grid = tf.stack(grid, axis=-1)
    grid = tf.cast(grid, 'float32')
    grid = tf.reshape(grid, (-1, 3))
    grid = tf_repeat_3d(grid, batch_size)
    coords = grid+offsets

    mapped_vals = tf_batch_map_coordinates(input, coords)
    return mapped_vals
