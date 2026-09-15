"""CS232 end-to-end demo: one command, the whole multi-user pipeline.

    Source video -> I/P/B classify -> H256 encode (cached per video_id)
    -> round-robin load balancer -> streaming nodes -> N independent
    users (each own bandwidth/buffer/history) -> H256 decode ->
    reconstruction -> simple SR -> per-user output

Usage:
    python demo.py                          run the whole pipeline once, print a summary
    python demo.py --video path/to/clip.mp4  use your own video instead of the sample clip
    python demo.py --users 4                 simulate 4 users watching the same video
    python demo.py --users 4 --user u2       print the frame list in the order sent to user u2
    python demo.py --frame 12 --user u2      print the 13-stage trace for frame 12 of user u2
    python demo.py --bandwidth 800           default bandwidth for every user (kbps)
    python demo.py --qp 16                   coarser quantization -> smaller, lossier bitstream
    python demo.py --web                     launch the browser dashboard instead of the CLI run
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pipeline
from codec.h256_encoder import DEFAULT_QP
from data.generate_sample import generate as generate_sample
from streaming.session import SessionManager, UserSession
from tools.visualize_trace import save_frame_outputs

DEFAULT_VIDEO = pipeline.DEFAULT_VIDEO


def _ensure_video(path: Path, num_frames: int) -> None:
    if not path.exists():
        print(f"[Setup] {path.name} not found, generating a synthetic sample clip...")
        generate_sample(path, num_frames=num_frames)


def _fmt_refs(*refs) -> str:
    vals = [str(r) for r in refs if r is not None]
    return ", ".join(vals) if vals else "-"


def print_user_summary(sessions: list[UserSession]) -> None:
    print()
    print(f"{'User':>6} {'Bandwidth':>10} {'Quality':>8} {'Frames':>7} {'Avg PSNR':>10} {'Avg SSIM':>9}")
    print("-" * 56)
    for s in sessions:
        snap = s.snapshot()
        ssim_values = [h["ssim"] for h in s.history]
        avg_psnr = f"{snap['avg_psnr']:.2f} dB" if snap["avg_psnr"] is not None else "inf"
        avg_ssim = f"{(sum(ssim_values) / len(ssim_values)):.4f}" if ssim_values else "-"
        print(
            f"{s.user_id:>6} {s.bandwidth_kbps:>9.0f}k {snap['quality'] or '-':>8} "
            f"{snap['frames_received']:>7} {avg_psnr:>10} {avg_ssim:>9}"
        )


def print_frame_table(session: UserSession) -> None:
    print()
    print(f"user {session.user_id} -- frames in the order they were received")
    print(f"{'#':>4} {'Frame':>5} {'Type':^5} {'Ref':^10} {'Quality':^7} {'Node':^8} {'Encoded (B)':>12} {'PSNR':>10}")
    print("-" * 72)
    for r in session.history:
        ref = _fmt_refs(*r["refs"])
        psnr_str = f"{r['psnr_db']:.2f} dB" if r["psnr_db"] is not None else "inf"
        print(
            f"{r['recv_order']:>4} {r['frame_id']:>5} {r['type']:^5} {ref:^10} {r['quality']:^7} "
            f"{r['node']:^8} {r['encoded_bytes']:>12} {psnr_str:>10}"
        )


def print_trace(trace: dict) -> None:
    print()
    print("=" * 50)
    print("END-TO-END TRACE")
    print("=" * 50)

    print(f"\nUser           : {trace['user_id']}")
    print("\n[1] SOURCE")
    print(f"Frame ID       : {trace['frame_id']}")
    print(f"Resolution     : {trace['source']['resolution']}")
    print(f"Original size  : {trace['source']['original_size_bytes']} bytes")

    print("\n[2] FRAME TYPE")
    print(f"Type           : {trace['frame_type']['type']}")
    print(f"Reference      : {_fmt_refs(trace['frame_type']['ref1'], trace['frame_type']['ref2'])}")

    print("\n[3] PREDICTION")
    print(f"Method         : {trace['prediction']['description']}")

    print("\n[4] RESIDUAL")
    print(f"Residual range : [{trace['residual']['min']}, {trace['residual']['max']}]")
    print(f"Residual MSE   : {trace['residual']['mse']:.2f}")

    print("\n[5] QUANTIZATION")
    print(f"QP             : {trace['quantization']['qp']}")

    print("\n[6] H256 ENCODE")
    print(f"Raw size       : {trace['h256_encode']['raw_size']} bytes")
    print(f"Encoded size   : {trace['h256_encode']['encoded_size']} bytes")
    print(f"Compression    : {trace['h256_encode']['compression_pct']:.1f}%")

    print("\n[7] BITSTREAM")
    print(f"Quality        : {trace['bitstream']['quality']}")
    print(f"Segment        : {trace['bitstream']['segment']}")

    print("\n[8] LOAD BALANCER")
    print(f"Request        : {trace['load_balancer']['request']}")
    print(f"Selected node  : {trace['load_balancer']['selected_node']}")
    print(f"Policy         : {trace['load_balancer']['policy']}   (counter={trace['load_balancer']['lb_counter']})")

    print("\n[9] CLIENT")
    print(f"Received bytes : {trace['client']['received_bytes']}")

    print("\n[10] H256 DECODE")
    print(f"Frame type     : {trace['h256_decode']['frame_type']}")
    print(f"Reference      : {_fmt_refs(trace['h256_decode']['ref1'], trace['h256_decode']['ref2'])}")

    print("\n[11] RECONSTRUCTION")
    print(f"Resolution     : {trace['reconstruction']['resolution']}")
    print(f"Reconstruction MSE  : {trace['reconstruction']['mse']:.2f}")
    print(f"Reconstruction PSNR : {trace['reconstruction']['psnr']:.2f} dB")
    print(f"Reconstruction SSIM : {trace['reconstruction']['ssim']:.4f}")

    print("\n[12] SUPER RESOLUTION")
    print(f"Input          : {trace['super_resolution']['input']}")
    print(f"Output         : {trace['super_resolution']['output']}")
    print(f"Method         : {trace['super_resolution']['method']}")

    out_dir = pipeline.user_dir_for(trace["video_id"], trace["user_id"]) / f"frame_{trace['frame_id']}"
    print("\n[13] FINAL")
    print(f"Output         : {out_dir / '05_sr.png'}")

    print("\n" + "=" * 50)
    print("TRACE COMPLETE")
    print("=" * 50)


def run_pipeline(args) -> None:
    _ensure_video(Path(args.video), args.frames)

    t0 = time.perf_counter()
    video_id = pipeline.video_id_of(Path(args.video))
    fps = pipeline.video_fps(Path(args.video))
    print(f"[1/7] Reading source video: {args.video}  (video_id={video_id})")
    native_frames = pipeline.load_source_video(Path(args.video), args.max_frames)
    num_frames = len(native_frames)
    print(f"      {num_frames} frames read (--max-frames={args.max_frames}, native fps={fps:.1f})")

    if args.frame is not None and not (0 <= args.frame < num_frames):
        print(f"error: --frame must be between 0 and {num_frames - 1}", file=sys.stderr)
        sys.exit(1)
    if args.users < 1:
        print("error: --users must be >= 1", file=sys.stderr)
        sys.exit(1)

    print("[2/7] Building ABR quality ladder (240p / 360p / 540p)")
    tiers = pipeline.build_quality_tiers(native_frames)

    print(f"[3/7] I/P/B classify + H256 encode/reuse (QP={args.qp})")
    infos, encode_traces = pipeline.ensure_encoded(video_id, tiers, qp=args.qp, log=lambda m: None)
    total_raw = sum(i.raw_size for i in infos if i.quality == "540p")
    total_enc = sum(i.encoded_size for i in infos if i.quality == "540p")
    print(f"      540p tier: {total_raw} -> {total_enc} bytes ({100 * (1 - total_enc / total_raw):.1f}% smaller)")

    print("[4/7] Starting streaming nodes + round-robin load balancer")
    handles = pipeline.start_streaming(log=print)

    print(f"[5/7] Creating {args.users} user(s) (bandwidth={args.bandwidth:.0f} kbps each)")
    sessions = SessionManager()
    users = [sessions.create_user(args.bandwidth) for _ in range(args.users)]

    target_user = args.user
    if target_user is not None and sessions.get(target_user) is None:
        # --user names someone outside the auto-numbered u1..uN range (e.g.
        # `python demo.py --frame 12 --user u2` on its own, no --users) --
        # provision them too rather than erroring, so every CLI example in
        # README section 2 works standalone.
        users.append(sessions.create_user(args.bandwidth, user_id=target_user))
    if target_user is None and len(users) == 1:
        target_user = users[0].user_id
    if args.frame is not None and target_user is None:
        print("error: --user is required (with --frame) when --users > 1", file=sys.stderr)
        handles.shutdown()
        sys.exit(1)

    try:
        print("[6/7] Streaming to every user in parallel via the load balancer")
        mv_index_cache: dict = {}
        pipeline.run_all_users(handles.lb_url, video_id, users, num_frames, fps, tiers, mv_index_cache, log=print)

        print("[7/7] Summary")
        print_user_summary(users)
        elapsed = time.perf_counter() - t0
        print(f"\nPipeline finished in {elapsed:.1f}s")

        if target_user is not None:
            print_frame_table(sessions.get(target_user))

        if args.frame is not None:
            print("\nTracing one frame")
            session = sessions.get(target_user)
            trace = pipeline.trace_user_frame(
                video_id, session, args.frame, native_frames, tiers, encode_traces, args.qp, mv_index_cache
            )
            print_trace(trace)
            out_dir = pipeline.user_dir_for(video_id, target_user) / f"frame_{args.frame}"
            save_frame_outputs(trace, out_dir)
            print(f"\nSaved trace images + trace.json + summary.png + mv_overlay.png -> {out_dir}")
    finally:
        handles.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="CS232 end-to-end streaming/codec/SR demo")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="source video (default: synthetic sample)")
    parser.add_argument("--frames", type=int, default=32, help="frame count if the sample clip needs generating")
    parser.add_argument(
        "--max-frames", type=int, default=pipeline.DEFAULT_MAX_FRAMES, help="cap on frames read from the source video"
    )
    parser.add_argument("--users", type=int, default=1, help="number of simulated concurrent users")
    parser.add_argument("--user", default=None, help="print this user's received-frame table")
    parser.add_argument("--frame", type=int, default=None, help="trace this single frame_id in detail (needs --user when --users > 1)")
    parser.add_argument("--bandwidth", type=float, default=pipeline.DEFAULT_BANDWIDTH_KBPS, help="bandwidth for every user (kbps)")
    parser.add_argument("--qp", type=int, default=DEFAULT_QP, help="quantization step (bigger = smaller/lossier)")
    parser.add_argument("--web", action="store_true", help="launch the browser dashboard instead")
    parser.add_argument("--port", type=int, default=7000, help="port for --web")
    args = parser.parse_args()

    if args.web:
        from web.app import run_dashboard

        run_dashboard(port=args.port)
        return

    run_pipeline(args)


if __name__ == "__main__":
    main()
