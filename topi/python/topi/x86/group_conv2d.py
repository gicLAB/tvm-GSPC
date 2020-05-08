import logging
import re

import tvm
from tvm import autotvm
from tvm import te
from .util import get_fp32_len
from ..util import get_const_tuple, simplify
from ..nn.pad import pad
from ..nn.util import infer_pad, get_pad_tuple
from .. import generic, tag

from ..nn.conv2d import group_conv2d_nchw
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity
from ..nn.conv2d import conv2d_infer_layout, _get_workload as _get_conv2d_workload
from .. import nn

def group_conv2d_nchw(data, kernel, strides, padding, dilation, groups, out_dtype):
    """Compute group_conv2d with NCHW layout"""
    return group_conv2d_nchwc_spatial_pack(data, kernel, strides, padding, dilation, groups, out_dtype)


def schedule_group_conv2d_nchw(outs):
    """Compute group_conv2d with NCHW layout"""
    return schedule_group_conv2d_nchwc(outs)


def _get_default_config(cfg, data, kernel, strides, padding, groups, out_dtype,
                        layout='NCHW'):
    """
    Get default schedule config for the workload
    """
    static_data_shape = []
    for dim in get_const_tuple(data.shape):
        if isinstance(dim, tvm.tir.Var):
            static_data_shape.append(1)
        else:
            static_data_shape.append(dim)
    data = te.placeholder(static_data_shape, dtype=data.dtype)

    wkl = _get_conv2d_workload(data, kernel, strides, padding, out_dtype, layout)
    is_kernel_1x1 = wkl.hkernel == 1 and wkl.wkernel == 1
    if is_kernel_1x1:
        conv2d_avx_1x1._fallback_schedule(cfg, wkl)
        raise NotImplementedError
    else:
        _fallback_schedule(cfg, wkl)


def _fallback_schedule(cfg, wkl):
    simd_width = get_fp32_len()
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride
    out_width = (wkl.width + 2 * WPAD - wkl.wkernel) // WSTR + 1
    G = wkl.groups
    KPG = wkl.out_filter // G
    CPG = wkl.in_filter // G
    oc_bn = 1

    # if KPG < simd_width:
    #     oc_bn = KPG
    # else:
    #     oc_bn = (KPG // simd_width) * simd_width
    oc_bn = 1
    for bn in range(simd_width, 0, -1):
        if KPG % bn == 0:
            oc_bn = bn
            break

    ic_bn = 1
    for bn in range(oc_bn, 0, -1):
        if CPG % bn == 0:
            ic_bn = bn
            break

    reg_n = 1
    for n in range(31, 0, -1):
        if out_width % n == 0:
            reg_n = n
            break

    cfg["tile_ic"] = SplitEntity([wkl.in_filter // ic_bn, ic_bn])
    cfg["tile_oc"] = SplitEntity([wkl.out_filter // oc_bn, oc_bn])
    cfg["tile_ow"] = SplitEntity([out_width // reg_n, reg_n])
    cfg["unroll_kw"] = OtherOptionEntity(False)


@autotvm.register_topi_compute("group_conv2d_nchw.x86")
def group_conv2d_nchwc_spatial_pack(cfg, data, kernel, strides, padding,
                                    dilation, groups, out_dtype='float32'):
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(dilation, int):
        dilation_h, dilation_w = dilation, dilation
    else:
        dilation_h, dilation_w = dilation

    assert isinstance(padding, int) or len(padding) == 2 or len(padding) == 4
    if isinstance(padding, int):
        HPAD, WPAD = padding, padding
    elif len(padding) == 2:
        HPAD, WPAD = padding
    else:
        HPAD, _, WPAD, _ = padding

    assert isinstance(strides, int) or len(strides) == 2
    if isinstance(strides, int):
        HSTR, WSTR = strides, strides
    else:
        HSTR, WSTR = strides

    batch_size, in_channel, in_height, in_width = get_const_tuple(data.shape)
    N, CI, IH, IW = get_const_tuple(data.shape)
    num_filter, _, kernel_height, kernel_width = get_const_tuple(kernel.shape)
    CO, CIG, KH, KW = get_const_tuple(kernel.shape)

    pad_height = in_height + 2 * HPAD
    pad_width = in_width + 2 * WPAD

    dilated_kernel_h = (kernel_height - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_width - 1) * dilation_w + 1
    out_height = (in_height + 2 * HPAD - dilated_kernel_h) // HSTR + 1
    out_width = (in_width + 2 * WPAD - dilated_kernel_w) // WSTR + 1

    OH = (in_height + 2 * HPAD - dilated_kernel_h) // HSTR + 1
    OW = (in_width + 2 * WPAD - dilated_kernel_w) // WSTR + 1

    simd_width = get_fp32_len()

    kpg = num_filter // groups # kernels per group

    G = groups
    KPG = num_filter // G
    CPG = in_channel // G

    cfg.define_split("tile_ic", in_channel, num_outputs=2)
    cfg.define_split("tile_oc", num_filter, num_outputs=2)
    cfg.define_split("tile_ow", OW, num_outputs=2, filter=lambda y: y.size[-1] <= 64)
    cfg.define_knob("unroll_kw", [True, False])

    # If no config was set, we can fallback to default config.
    if cfg.is_fallback:
        _get_default_config(cfg, te.placeholder((N, CI, IH, IW), dtype=data.dtype),
                            te.placeholder((CO, CIG, KH, KW),
                                           dtype=kernel.dtype),
                            strides, padding, groups, out_dtype)

    oc_bn = cfg['tile_oc'].size[-1]
    ic_bn = cfg['tile_ic'].size[-1]
    # pack data
    DOPAD = (HPAD != 0 or WPAD != 0)
    if DOPAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")
    else:
        data_pad = data

    batch_size, in_channel, in_height, in_width = get_const_tuple(data.shape)
    num_filter, _, kernel_height, kernel_width = get_const_tuple(kernel.shape)

    shape = (groups, batch_size, CPG // ic_bn,
             pad_height, ic_bn, pad_width)

    data_vec = te.compute(shape,
                          lambda g, n, C, h, c, w:
                          data_pad[n, C * ic_bn + c + CPG * g, h, w],
                          name='data_vec')

    # pack kernel
    shape = (groups, KPG//oc_bn, CPG//ic_bn,
             kernel_height, kernel_width, ic_bn, oc_bn)
    kernel_vec = te.compute(shape,
                            lambda g, CO, CI, h, w, ci, co:
                            kernel[(CO * oc_bn + co + g * KPG),
                                   CI * ic_bn + ci, h, w],
                            name='kernel_vec')

    # convolution
    oshape = (groups, batch_size, KPG//oc_bn,
              out_height, out_width, oc_bn)
    unpack_shape = (batch_size, num_filter, out_height, out_width)

    ic = te.reduce_axis((0, (CPG)), name='ic')
    kh = te.reduce_axis((0, kernel_height), name='kh')
    kw = te.reduce_axis((0, kernel_width), name='kw')
    idxmod = tvm.tir.indexmod
    idxdiv = tvm.tir.indexdiv

    conv = te.compute(oshape, lambda g, n, oc_chunk, oh, ow, oc_block:
                      te.sum(data_vec[g, n, idxdiv(ic, ic_bn),
                                      oh*HSTR+kh*dilation_h,
                                      idxmod(ic, ic_bn),
                                      ow*WSTR+kw*dilation_w].astype(out_dtype) *
                             kernel_vec[g, oc_chunk, idxdiv(ic, ic_bn),
                                        kh, kw, idxmod(ic, ic_bn),
                                        oc_block].astype(out_dtype),
                             axis=[ic, kh, kw]), name='conv')

    unpack = te.compute(unpack_shape,
                        lambda n, c, h, w:
                        conv[idxdiv(c, KPG), n,
                             idxmod(idxdiv(c, oc_bn), (KPG // oc_bn)),
                             h, w,
                             idxmod(idxmod(c, oc_bn), KPG)]
                        .astype(out_dtype),
                        name='output_unpack',
                        tag='group_conv2d_nchw')
    return unpack


@autotvm.register_topi_schedule("group_conv2d_nchw.x86")
def schedule_group_conv2d_nchwc(cfg, outs):
    """Create schedule for tensors"""
    s = te.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.te.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if 'group_conv2d_nchw_direct' in op.tag:
            output = op.output(0)
            schedule_conv_sp_direct(s, cfg, output)
        elif 'group_conv2d_nchw' in op.tag:
            output = op.output(0)

            if "tile_ic" not in cfg:
                return
            conv_out = op.input_tensors[0]
            kernel_vec = conv_out.op.input_tensors[1]
            kernel = kernel_vec.op.input_tensors[0]
            if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()
            data_vec = conv_out.op.input_tensors[0]
            data = data_vec.op.input_tensors[0]
            data_pad = None
            if isinstance(data.op, tvm.te.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            _, c, h, w = get_const_tuple(data.shape)
            _, x, kh, kw = get_const_tuple(kernel.shape)

            is_kernel_1x1 = kh == 1 and kw == 1
            args = [s, cfg, data, data_pad, data_vec, kernel_vec, conv_out, output, outs[0]]
            if is_kernel_1x1:
                raise NotImplementedError
            else:
                schedule_conv_sp_grouped(*args)

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s


def schedule_conv_sp_grouped(s, cfg, data, data_pad, data_vec, kernel_vec, conv_out, output, last,
                             **kwargs):
    # fetch schedule
    ic_bn, oc_bn, reg_n, unroll_kw = (cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1],
                                      cfg["tile_ow"].size[-1], cfg["unroll_kw"].val)

    # no stride and padding info here
    padding = infer_pad(data, data_pad)
    HPAD, WPAD = padding
    DOPAD = (HPAD != 0 or WPAD != 0)

    A, W = data, kernel_vec
    A0, A1 = data_pad, data_vec

    # schedule data
    if DOPAD:
        s[A0].compute_inline()
    groups, batch, ic_chunk, ih, ic_block, iw = s[A1].op.axis

    parallel_axis = s[A1].fuse(batch, ic_chunk, ih)
    s[A1].parallel(parallel_axis)

    # schedule kernel pack
    groups, oc_chunk, ic_chunk, oh, ow, ic_block, oc_block = s[W].op.axis
    s[W].reorder(oc_chunk, oh, ic_chunk, ow, ic_block, oc_block)

    if oc_bn > 1:
        s[W].vectorize(oc_block)

    parallel_axis = s[W].fuse(groups, oc_chunk, oh)
    s[W].parallel(parallel_axis)

    # schedule conv
    C, O0, O = conv_out, output, last
    CC = s.cache_write(C, 'global')

    _, _, oc_chunk, oh, ow, oc_block = s[C].op.axis

    ow_chunk, ow_block = s[C].split(ow, factor=reg_n)

    s[C].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    s[C].fuse(oc_chunk, oh)
    s[C].vectorize(oc_block)

    # if use_others:
    #     s[CC].compute_at(s[C], ow_chunk)
    groups, batch, oc_chunk, oh, ow, oc_block = s[CC].op.axis

    ic, kh, kw = s[CC].op.reduce_axis
    ow_chunk, ow_block = s[CC].split(ow, factor=reg_n)
    ic_chunk, ic_block = s[CC].split(ic, factor=ic_bn)

    if unroll_kw:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, ic_block, kw, ow_block, oc_block)
        s[CC].unroll(kw)
    else:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, kw, ic_block, ow_block, oc_block)


#    s[CC].fuse(oc_chunk, oh)
    parallel_axis = s[CC].fuse(groups, batch, oc_chunk, oh)
    s[CC].parallel(parallel_axis)

    s[CC].vectorize(oc_block)

    s[CC].unroll(ow_block)


    if O0 != O:
        s[O0].compute_inline()

    batch, oc, oh, ow = s[O].op.axis
    ow_chunk, ow_block = s[O].split(ow, factor=reg_n)
    oc_chunk, oc_block = s[O].split(oc, factor=oc_bn)

    s[O].reorder(batch, oc_chunk, oh, ow_chunk, ow_block, oc_block)
    parallel_axis = s[O].fuse(oc_chunk, oh)
    #s[C].compute_at(s[O], parallel_axis)
    s[O].vectorize(oc_block)
    s[O].parallel(parallel_axis)
    return s


@autotvm.register_topi_compute("group_conv2d_nchw.direct.x86")
def group_conv2d_nchw_direct(cfg, Input, Filter, stride, padding, dilation, groups, out_dtype=None):
    """Group convolution operator in NCHW layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.te.Tensor
        4-D with shape [num_filter, in_channel // groups, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation : int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    groups : int
        number of groups

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_channel, in_height, in_width = get_const_tuple(Input.shape)
    num_filter, _, kernel_h, kernel_w = get_const_tuple(Filter.shape)

    assert in_channel % groups == 0, "input channels must divide group size"
    assert num_filter % groups == 0, "output channels must divide group size"

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (kernel_h, kernel_w))
    # compute the output shape
    out_channel = num_filter
    out_height = simplify(
        (in_height - (kernel_h - 1) * dilation_h - 1 + pad_top + pad_down) // stride_h + 1)
    out_width = simplify(
        (in_width - (kernel_w - 1) * dilation_w - 1 + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    temp = pad(Input, pad_before, pad_after, name="pad_temp")
    rc = te.reduce_axis((0, in_channel // groups), name='rc')
    ry = te.reduce_axis((0, kernel_h), name='ry')
    rx = te.reduce_axis((0, kernel_w), name='rx')

    return te.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, ff, yy, xx: te.sum(
            temp[nn, ff // (num_filter//groups) * (in_channel//groups) + rc,
                 yy * stride_h + ry * dilation_h,
                 xx * stride_w + rx * dilation_w].astype(out_dtype) *
            Filter[ff, rc, ry, rx].astype(out_dtype),
            axis=[rc, ry, rx]), tag='group_conv2d_nchw_direct')

def schedule_conv_sp_direct(s, cfg, output):
    C = output
    batch, oc, oh, ow = s[C].op.axis
    s[C].parallel(oc)
    return s
