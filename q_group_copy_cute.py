from __future__ import annotations

import argparse
import math

import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync

from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned, to_cute_tensor
from flash_attn.cute import utils


class QGroupCopyExperiment:
    def __init__(
        self,
        *,
        qhead: int = 16,
        head_dim: int = 128,
        m_block: int = 128,
        num_threads: int = 128,
        mode: str = "row",
        layout_kind: str = "mma",
        copy_bits: int = 128,
        group_value_layout: str = "simple",
    ):
        self.qhead = qhead
        self.head_dim = head_dim
        self.m_block = m_block
        self.num_threads = num_threads
        self.mode = mode
        self.layout_kind = layout_kind
        self.copy_bits = copy_bits
        self.group_value_layout = group_value_layout
        self.buffer_align_bytes = 1024

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mIdx: cute.Tensor,
        mOut: cute.Tensor,
        row_start: Int32,
        n_rows: Int32,
        sQ_layout,
        gmem_tiled_copy_row: cute.TiledCopy,
        gmem_tiled_copy_group: cute.TiledCopy,
        gmem_tiled_copy_flat: cute.TiledCopy,
        gmem_tiled_copy_direct: cute.TiledCopy,
        gmem_tiled_copy_cotiled: cute.TiledCopy,
        gmem_tiled_copy_out: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage)
        if const_expr(self.layout_kind == "mma"):
            sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
            sQ_stage = cute.make_tensor(
                sQ.iterator,
                cute.make_layout(
                    (sQ.shape[0][0], (sQ.shape[0][1], sQ.shape[2])),
                    stride=(sQ.stride[0][0], (sQ.stride[0][1], sQ.stride[2])),
                ),
            )
        else:
            sQ_stage = storage.sQ.get_tensor(sQ_layout)
        qhead = const_expr(self.qhead)
        head_dim = const_expr(self.head_dim)

        if const_expr(self.mode == "attn_group"):
            group_copy = gmem_tiled_copy_group.get_slice(tidx)
            row_copy = gmem_tiled_copy_row.get_slice(tidx)
            cQ = cute.make_identity_tensor((self.m_block, self.head_dim))
            tQsQ = row_copy.partition_D(sQ_stage)
            tQcQ = row_copy.partition_S(cQ)
            elems_per_load = cute.size(tQsQ.shape[0][0])
            group_cQ = cute.make_identity_tensor((qhead, head_dim))
            q_groups = const_expr(self.m_block // qhead)
            sQ_group = cute.make_tensor(
                sQ_stage.iterator,
                cute.make_layout(
                    (q_groups, (qhead, sQ_stage.shape[1])),
                    stride=(
                        qhead * sQ_stage.stride[0],
                        (sQ_stage.stride[0], sQ_stage.stride[1]),
                    ),
                ),
            )
            for group in cutlass.range_constexpr(q_groups):
                row0 = group * qhead
                virtual0 = row_start + row0
                group_aligned = (virtual0 % qhead) == 0
                if row_start % qhead == 0 and group_aligned and row0 < n_rows:
                    qidx = Int32(mIdx[virtual0 // qhead])
                    head0 = (virtual0 % qhead)
                    q_ptr = cute.make_ptr(
                        mQ.element_type,
                        utils.elem_pointer(mQ, (qidx, head0, 0)).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    src = cute.make_tensor(
                        q_ptr,
                        cute.make_layout((qhead, head_dim), stride=(head_dim, 1)),
                    )
                    tSrc = group_copy.partition_S(src)
                    tDst = group_copy.partition_D(sQ_group[group, (None, None)])
                    pred = cute.make_fragment((2, (1, 1)), cutlass.Boolean)
                    local_h = tidx // (self.num_threads // qhead)
                    row_valid = row0 + local_h < n_rows
                    for i in cutlass.range_constexpr(cute.size(pred)):
                        pred[i] = row_valid
                    cute.copy(group_copy, tSrc, tDst, pred=pred)
                else:
                    pass

            t0QcQ = gmem_tiled_copy_row.get_slice(0).partition_S(cQ)
            tQcQ_row = tQcQ[0, None, 0]
            threads_per_row = gmem_tiled_copy_row.layout_tv_tiled.shape[0][0]
            num_ptr = cute.ceil_div(cute.size(tQcQ_row), threads_per_row)
            tPrPtr = cute.make_fragment(num_ptr, Int64)
            tPrNeedCopy = cute.make_fragment(num_ptr, Int32)
            for i in cutlass.range_constexpr(num_ptr):
                row = i * self.num_threads + tQcQ_row[tidx % threads_per_row][0]
                head = Int32(0)
                qidx = Int32(0)
                need_copy = Int32(0)
                if row < n_rows:
                    virtual = row_start + row
                    ql = virtual // qhead
                    head = virtual % qhead
                    qidx = Int32(mIdx[ql])
                    local_group_start = ql * qhead - row_start
                    need_copy = (
                        Int32(1)
                        if row_start % qhead != 0
                        else Int32(0)
                    )
                tPrPtr[i] = utils.elem_pointer(mQ, (qidx, head, 0)).toint()
                tPrNeedCopy[i] = need_copy

            for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
                need_copy = utils.shuffle_sync(
                    tPrNeedCopy[m // threads_per_row], m % threads_per_row, width=threads_per_row
                )
                q_ptr_i64 = utils.shuffle_sync(
                    tPrPtr[m // threads_per_row], m % threads_per_row, width=threads_per_row
                )
                q_ptr = cute.make_ptr(
                    mQ.element_type,
                    q_ptr_i64,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                if need_copy != 0:
                    src = cute.make_tensor(q_ptr, (head_dim,))
                    src_copy = cute.tiled_divide(src, (elems_per_load,))
                    for k in cutlass.range_constexpr(cute.size(tQsQ.shape[2])):
                        ki = tQcQ[0, 0, k][1] // elems_per_load
                        cute.copy(row_copy, src_copy[None, ki], tQsQ[None, m, k])
        elif const_expr(self.mode == "cotiled"):
            cotiled_copy = gmem_tiled_copy_cotiled.get_slice(tidx)
            q_groups = const_expr(self.m_block // qhead)
            sQ_group = cute.make_tensor(
                sQ_stage.iterator,
                cute.make_layout(
                    (q_groups, (qhead, sQ_stage.shape[1])),
                    stride=(
                        qhead * sQ_stage.stride[0],
                        (sQ_stage.stride[0], sQ_stage.stride[1]),
                    ),
                ),
            )
            for group in cutlass.range_constexpr(q_groups):
                qidx = Int32(mIdx[group])
                q_ptr = cute.make_ptr(
                    mQ.element_type,
                    utils.elem_pointer(mQ, (qidx, 0, 0)).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                src = cute.make_tensor(
                    q_ptr,
                    cute.make_layout((qhead, head_dim), stride=(head_dim, 1)),
                )
                tSrc = cotiled_copy.partition_S(src)
                tDst = cotiled_copy.partition_D(sQ_group[group, (None, None)])
                cute.copy(cotiled_copy, tSrc, tDst)
        elif const_expr(self.mode == "direct"):
            direct_copy = gmem_tiled_copy_direct.get_slice(tidx)
            q_groups = const_expr(self.m_block // qhead)
            sQ_group = cute.make_tensor(
                sQ_stage.iterator,
                cute.make_layout(
                    (q_groups, (qhead, sQ_stage.shape[1])),
                    stride=(
                        qhead * sQ_stage.stride[0],
                        (sQ_stage.stride[0], sQ_stage.stride[1]),
                    ),
                ),
            )
            for group in cutlass.range_constexpr(q_groups):
                qidx = Int32(mIdx[group])
                q_ptr = cute.make_ptr(
                    mQ.element_type,
                    utils.elem_pointer(mQ, (qidx, 0, 0)).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                src = cute.make_tensor(
                    q_ptr,
                    cute.make_layout((qhead, head_dim), stride=(head_dim, 1)),
                )
                tSrc = direct_copy.partition_S(src)
                tDst = direct_copy.partition_D(sQ_group[group, (None, None)])
                cute.copy(direct_copy, tSrc, tDst)
        elif const_expr(self.mode == "group_modes"):
            flat_copy = gmem_tiled_copy_flat.get_slice(tidx)
            q_groups = const_expr(self.m_block // qhead)
            sQ_group = cute.make_tensor(
                sQ_stage.iterator,
                cute.make_layout(
                    (q_groups, (qhead, sQ_stage.shape[1])),
                    stride=(
                        qhead * sQ_stage.stride[0],
                        (sQ_stage.stride[0], sQ_stage.stride[1]),
                    ),
                ),
            )
            for group in cutlass.range_constexpr(q_groups):
                qidx = Int32(mIdx[group])
                q_ptr = cute.make_ptr(
                    mQ.element_type,
                    utils.elem_pointer(mQ, (qidx, 0, 0)).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                src = cute.make_tensor(q_ptr, (qhead * head_dim,))
                dst = cute.group_modes(sQ_group[group, (None, None)], 0, 2)
                tSrc = flat_copy.partition_S(src)
                tDst = flat_copy.partition_D(dst)
                cute.copy(flat_copy, tSrc, tDst)
        elif const_expr(self.mode == "flat"):
            flat_copy = gmem_tiled_copy_flat.get_slice(tidx)
            q_groups = const_expr(self.m_block // qhead)
            sQ_flat = cute.make_tensor(
                sQ_stage.iterator,
                cute.make_layout(
                    (q_groups, qhead * head_dim),
                    stride=(qhead * sQ_stage.stride[0], 1),
                ),
            )
            for group in cutlass.range_constexpr(q_groups):
                qidx = Int32(mIdx[group])
                q_ptr = cute.make_ptr(
                    mQ.element_type,
                    utils.elem_pointer(mQ, (qidx, 0, 0)).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                src = cute.make_tensor(q_ptr, (qhead * head_dim,))
                tSrc = flat_copy.partition_S(src)
                tDst = flat_copy.partition_D(sQ_flat[group, None])
                cute.copy(flat_copy, tSrc, tDst)
        elif const_expr(self.mode == "group"):
            group_copy = gmem_tiled_copy_group.get_slice(tidx)
            q_groups = const_expr(self.m_block // qhead)
            sQ_group = cute.make_tensor(
                sQ_stage.iterator,
                cute.make_layout(
                    (q_groups, (qhead, sQ_stage.shape[1])),
                    stride=(
                        qhead * sQ_stage.stride[0],
                        (sQ_stage.stride[0], sQ_stage.stride[1]),
                    ),
                ),
            )
            for group in cutlass.range_constexpr(q_groups):
                qidx = Int32(mIdx[group])
                q_ptr = cute.make_ptr(
                    mQ.element_type,
                    utils.elem_pointer(mQ, (qidx, 0, 0)).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                src = cute.make_tensor(
                    q_ptr,
                    cute.make_layout((qhead, head_dim), stride=(head_dim, 1)),
                )
                tSrc = group_copy.partition_S(src)
                tDst = group_copy.partition_D(sQ_group[group, (None, None)])
                cute.copy(group_copy, tSrc, tDst)
        else:
            row_copy = gmem_tiled_copy_row.get_slice(tidx)
            cQ = cute.make_identity_tensor((self.m_block, self.head_dim))
            tQsQ = row_copy.partition_D(sQ_stage)
            tQcQ = row_copy.partition_S(cQ)
            t0QcQ = gmem_tiled_copy_row.get_slice(0).partition_S(cQ)
            tQcQ_row = tQcQ[0, None, 0]
            threads_per_row = gmem_tiled_copy_row.layout_tv_tiled.shape[0][0]
            num_ptr = cute.ceil_div(cute.size(tQcQ_row), threads_per_row)
            tPrPtr = cute.make_fragment(num_ptr, Int64)
            for i in cutlass.range_constexpr(num_ptr):
                row = i * self.num_threads + tQcQ_row[tidx % threads_per_row][0]
                qidx = Int32(mIdx[row // qhead])
                head = row % qhead
                tPrPtr[i] = utils.elem_pointer(mQ, (qidx, head, 0)).toint()

            elems_per_load = cute.size(tQsQ.shape[0][0])
            for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
                q_ptr_i64 = utils.shuffle_sync(
                    tPrPtr[m // threads_per_row], m % threads_per_row, width=threads_per_row
                )
                q_ptr = cute.make_ptr(
                    mQ.element_type,
                    q_ptr_i64,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                src = cute.make_tensor(q_ptr, (head_dim,))
                src_copy = cute.tiled_divide(src, (elems_per_load,))
                for k in cutlass.range_constexpr(cute.size(tQsQ.shape[2])):
                    ki = tQcQ[0, 0, k][1] // elems_per_load
                    cute.copy(row_copy, src_copy[None, ki], tQsQ[None, m, k])

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        out_copy = gmem_tiled_copy_out.get_slice(tidx)
        tOsQ = out_copy.partition_S(sQ_stage)
        tOgO = out_copy.partition_D(mOut)
        cute.copy(gmem_tiled_copy_out, tOsQ, tOgO)

    @cute.jit
    def __call__(self, mQ, mIdx, mOut, row_start: Int32, n_rows: Int32, stream: cuda.CUstream = None):
        mQ, mOut = [assume_tensor_aligned(t) for t in (mQ, mOut)]
        groups = const_expr(self.m_block // self.qhead)
        mQ = cute.make_tensor(
            mQ.iterator,
            cute.make_layout((groups, self.qhead, self.head_dim), stride=(self.qhead * self.head_dim, self.head_dim, 1)),
        )
        mIdx = cute.make_tensor(mIdx.iterator, cute.make_layout((groups,), stride=(1,)))
        mOut = cute.make_tensor(
            mOut.iterator,
            cute.make_layout((self.m_block, self.head_dim), stride=(self.head_dim, 1)),
        )
        dtype = mQ.element_type
        if const_expr(self.layout_kind == "mma"):
            tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
                dtype,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.K,
                Float32,
                tcgen05.CtaGroup.ONE,
                (self.m_block, self.head_dim),
            )
            mma_tiler_qk = (self.m_block, self.head_dim, self.head_dim)
            sQ_layout = sm100_utils_basic.make_smem_layout_a(tiled_mma_qk, mma_tiler_qk, dtype, 1)
        else:
            sQ_layout = cute.make_layout((self.m_block, self.head_dim), stride=(self.head_dim, 1))

        universal_copy_bits = const_expr(self.copy_bits)
        async_copy_elems = universal_copy_bits // dtype.width
        atom_async = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        threads_per_row = math.gcd(self.head_dim // async_copy_elems, self.num_threads)
        row_thr_layout = cute.make_ordered_layout(
            (self.num_threads // threads_per_row, threads_per_row),
            order=(1, 0),
        )
        row_val_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_row = cute.make_tiled_copy_tv(atom_async, row_thr_layout, row_val_layout)

        group_threads_per_head = self.num_threads // self.qhead
        group_thr_layout = cute.make_ordered_layout(
            (self.qhead, group_threads_per_head),
            order=(1, 0),
        )
        if const_expr(self.group_value_layout == "simple"):
            group_val_layout = cute.make_layout((1, 2 * async_copy_elems), stride=(0, 1))
        elif const_expr(self.group_value_layout == "nested"):
            group_val_layout = cute.make_layout(
                ((1, 2), async_copy_elems),
                stride=((0, group_threads_per_head * async_copy_elems), 1),
            )
        else:
            group_val_layout = cute.make_layout((1, async_copy_elems), stride=(0, 1))
        gmem_tiled_copy_group = cute.make_tiled_copy_tv(atom_async, group_thr_layout, group_val_layout)

        flat_thr_layout = cute.make_layout(self.num_threads)
        flat_val_layout = cute.make_layout(async_copy_elems)
        gmem_tiled_copy_flat = cute.make_tiled_copy_tv(atom_async, flat_thr_layout, flat_val_layout)

        direct_layout_tv = cute.make_layout(
            ((group_threads_per_head, self.qhead), 2),
            stride=((self.qhead * async_copy_elems, 1), self.qhead * group_threads_per_head * async_copy_elems),
        )
        gmem_tiled_copy_direct = cute.make_tiled_copy(
            atom_async,
            direct_layout_tv,
            (self.qhead, self.head_dim),
        )
        cotiled_atom_layout_tv = cute.make_layout(
            ((group_threads_per_head, self.qhead), 2 * async_copy_elems),
            stride=((async_copy_elems, self.head_dim), 1),
        )
        cotiled_data_layout = cute.make_layout(
            (self.qhead, self.head_dim),
            stride=(self.head_dim, 1),
        )
        gmem_tiled_copy_cotiled = cute.make_cotiled_copy(
            atom_async,
            cotiled_atom_layout_tv,
            cotiled_data_layout,
        )

        atom_out = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=universal_copy_bits)
        gmem_tiled_copy_out = cute.make_tiled_copy_tv(atom_out, row_thr_layout, row_val_layout)

        sQ_size = cute.cosize(sQ_layout)

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[cute.struct.MemRange[dtype, sQ_size], self.buffer_align_bytes]

        self.shared_storage = SharedStorage
        self.kernel(
            mQ,
            mIdx,
            mOut,
            row_start,
            n_rows,
            sQ_layout,
            gmem_tiled_copy_row,
            gmem_tiled_copy_group,
            gmem_tiled_copy_flat,
            gmem_tiled_copy_direct,
            gmem_tiled_copy_cotiled,
            gmem_tiled_copy_out,
        ).launch(grid=[1, 1, 1], block=[self.num_threads, 1, 1], smem=SharedStorage.size_in_bytes(), stream=stream)


_compile_cache: dict[tuple, object] = {}


def _run(
    mode: str,
    *,
    layout_kind: str,
    qhead: int,
    groups: int,
    iters: int,
    copy_bits: int,
    group_value_layout: str,
    row_start: int,
    n_rows: int | None,
) -> None:
    device = "cuda"
    dtype = torch.bfloat16
    head_dim = 128
    m_block = qhead * groups
    assert m_block == 128, "this scratch kernel currently assumes one 128-row tile"
    if n_rows is None:
        n_rows = m_block
    assert 0 <= n_rows <= m_block
    q_groups_needed = (row_start + n_rows + qhead - 1) // qhead
    q = torch.randn(q_groups_needed, qhead, head_dim, device=device, dtype=dtype)
    idx = torch.arange(q_groups_needed, device=device, dtype=torch.int32)
    out = torch.empty(m_block, head_dim, device=device, dtype=dtype)
    q_t = to_cute_tensor(q, leading_dim=2)
    idx_t = to_cute_tensor(idx, assumed_align=4, leading_dim=0)
    out_t = to_cute_tensor(out, leading_dim=1)
    key = (mode, layout_kind, qhead, groups, dtype, copy_bits, group_value_layout, row_start, n_rows)
    if key not in _compile_cache:
        kernel = QGroupCopyExperiment(
            qhead=qhead,
            m_block=m_block,
            mode=mode,
            layout_kind=layout_kind,
            copy_bits=copy_bits,
            group_value_layout=group_value_layout,
        )
        _compile_cache[key] = cute.compile(
            kernel,
            q_t,
            idx_t,
            out_t,
            row_start,
            n_rows,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    compiled = _compile_cache[key]
    compiled(q, idx, out, row_start, n_rows)
    torch.cuda.synchronize()
    ref = torch.empty(n_rows, head_dim, device=device, dtype=dtype)
    for row in range(n_rows):
        virtual = row_start + row
        ref[row] = q[int(idx[virtual // qhead].item()), virtual % qhead]
    max_err = (out[:n_rows].float() - ref.float()).abs().max().item() if n_rows > 0 else 0.0
    print(
        f"mode={mode} layout={layout_kind} qhead={qhead} "
        f"copy_bits={copy_bits} group_value_layout={group_value_layout} "
        f"row_start={row_start} n_rows={n_rows} max_err={max_err:.3e}"
    )
    if max_err != 0.0:
        raise SystemExit("copy mismatch")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        compiled(q, idx, out, row_start, n_rows)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        compiled(q, idx, out, row_start, n_rows)
    end.record()
    end.synchronize()
    print(
        f"mode={mode} layout={layout_kind} copy_bits={copy_bits} "
        f"group_value_layout={group_value_layout} avg_us={start.elapsed_time(end) * 1000.0 / iters:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["row", "group", "flat", "group_modes", "direct", "cotiled", "attn_group"],
        default="row",
    )
    parser.add_argument("--layout", choices=["plain", "mma"], default="mma")
    parser.add_argument("--copy-bits", type=int, choices=[16, 32, 64, 128], default=128)
    parser.add_argument("--group-value-layout", choices=["simple", "nested", "single"], default="simple")
    parser.add_argument("--qhead", type=int, default=16)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--n-rows", type=int, default=None)
    parser.add_argument("--iters", type=int, default=1000)
    args = parser.parse_args()
    _run(
        args.mode,
        layout_kind=args.layout,
        qhead=args.qhead,
        groups=args.groups,
        iters=args.iters,
        copy_bits=args.copy_bits,
        group_value_layout=args.group_value_layout,
        row_start=args.row_start,
        n_rows=args.n_rows,
    )


if __name__ == "__main__":
    main()
