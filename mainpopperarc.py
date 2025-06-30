import argparse
import os
import random
from pathlib import Path
import time
import traceback
from typing import Any, Dict, List, Tuple
import shutil
import concurrent.futures
import psutil
import logging

from init.init import prepare_arc_data, get_test_pairs
from bkbias.objattr import (
    determine_background_color,
    generate_files_from_task,
    nonbg_pixels,
    run_popper_from_dir,
    generate_test_bk,
    predict_from_prolog,
    evaluate_prediction,
    save_grid_txt,
    print_grid,
)


# Enable predicate invention by default
DEFAULT_ENABLE_PI = True


def default_worker_count(reserve: int = 3) -> int:
    """Return default number of parallel workers."""
    total = psutil.cpu_count(logical=False) or 1
    return max(1, total - reserve)


def count_non_background_pixels(task_data: Dict[str, Any], pixel_threshold_pct: int) -> int:
    """Return the total number of non-background pixels in all train grids."""
    bg_color = determine_background_color(task_data, pixel_threshold_pct=pixel_threshold_pct, debug=False)
    count = 0
    for pair in task_data.get("train", []):
        for kind in ("input", "output"):
            grid: List[List[int]] = pair.get(kind, [])
            count += len(nonbg_pixels(grid, bg_color))
    return count


def solve_task(
    task_id: str,
    task_data: Dict[str, Any],
    *,
    output_base: str,
    use_pixels: bool,
    bk_use_pixels: bool | None,
    exs_use_pixels: bool | None,
    enable_pi: bool,
    bg_threshold: int,
    popper_debug: bool = True,
    show_stdout: bool = True,
) -> Tuple[bool, str | None]:
    """Generate Popper files for a task and run the solver."""
    out_dir = os.path.join(output_base, task_id)
    try:
        bg_color = determine_background_color(
            task_data, pixel_threshold_pct=bg_threshold, debug=True
        )
        for idx, pair in enumerate(task_data.get("train", [])):
            print_grid(pair.get("input"), f"Train {task_id} input {idx}")
            print_grid(pair.get("output"), f"Train {task_id} output {idx}")
        bk_path, bias_path, exs_path = generate_files_from_task(
            task_data,
            out_dir,
            use_pixels=use_pixels,
            bk_use_pixels=bk_use_pixels,
            exs_use_pixels=exs_use_pixels,
            enable_pi=enable_pi,
            pixel_threshold_pct=bg_threshold,
            background_color=bg_color,
        )
        # for label, path in ("BK", bk_path), ("Bias", bias_path), ("Examples", exs_path):
        #     print(f"\n============================================== {label} for {task_id} =======================")
        #     with open(path) as f:
        #         print(f.read())
        prog_path, info = run_popper_from_dir(out_dir, debug=popper_debug, show_output=show_stdout)
        score = info.get("score")
        if prog_path is not None:
            print(f"！！！！！！！！！！！！！！！！！！！！！！Solved {task_id} with score {score}")
            hyp_path = os.path.join(out_dir, "hyp.pl")
            shutil.copy(prog_path, hyp_path)
            return True, hyp_path
        else:
            print(f"No solution for {task_id}")
            return False, None
    except Exception as e:  # pragma: no cover - runtime errors
        print(f"Error processing {task_id}: {e}")
        traceback.print_exc()
        return False, None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run Popper on ARC tasks")
    parser.add_argument(
        "--repr",
        choices=["pixels", "objects"],
        default="pixels",
        help="Global representation (pixels or objects).",
    )
    parser.add_argument(
        "--bk-repr",
        choices=["pixels", "objects"],
        default=None,
        help="Override BK representation",
    )
    parser.add_argument(
        "--exs-repr",
        choices=["pixels", "objects"],
        default=None,
        help="Override EXS representation",
    )
    parser.add_argument(
        "--enable-pi",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_PI,
        help="Enable predicate invention (hole2color).",
    )
    parser.add_argument(
        "--out",
        default="popper_kb",
        help="Directory to store generated files",
    )
    parser.add_argument(
        "--bg-threshold",
        type=int,
        default=40,
        help="Background color detection threshold percentage",
    )
    parser.add_argument(
        "--task-id",
        # default="0a2355a6",
        help="Run only the specified ARC task id",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--popper-debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run Popper in debug mode",
    )
    args = parser.parse_args()

    if args.popper_debug:
        logging.getLogger().setLevel(logging.DEBUG)

    use_pixels = args.repr == "pixels"
    bk_use_pixels = None if args.bk_repr is None else args.bk_repr == "pixels"
    exs_use_pixels = None if args.exs_repr is None else args.exs_repr == "pixels"

    train_tasks, train_sols, eval_tasks, eval_sols, test_tasks = prepare_arc_data()
    task_counts: List[Tuple[str, int]] = []
    for tid, tdata in train_tasks.items():
        cnt = count_non_background_pixels(tdata, args.bg_threshold)
        task_counts.append((tid, cnt))
    task_counts.sort(key=lambda x: x[1])

    os.makedirs(args.out, exist_ok=True)

    success_count = 0
    if args.task_id:
        selected = [(tid, cnt) for tid, cnt in task_counts if tid == args.task_id]
    else:
        selected = task_counts

    total_tasks = len(selected)
    if total_tasks == 1:
        workers = 1
    else:
        workers = args.workers

    show_stdout_tid = random.choice(selected)[0] if selected else None

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        future_map = {}
        for idx, (tid, _) in enumerate(selected, start=1):
            print(
                " * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * "
            )
            print(f"Queueing {tid} ({idx}/{total_tasks})")
            fut = ex.submit(
                solve_task,
                tid,
                train_tasks[tid],
                output_base=args.out,
                use_pixels=use_pixels,
                bk_use_pixels=bk_use_pixels,
                exs_use_pixels=exs_use_pixels,
                enable_pi=args.enable_pi,
                bg_threshold=args.bg_threshold,
                popper_debug=args.popper_debug,
                show_stdout=(tid == show_stdout_tid),
            )
            future_map[fut] = tid

        for fut in concurrent.futures.as_completed(future_map):
            tid = future_map[fut]
            try:
                solved, hyp = fut.result()
            except Exception as e:
                print(f"Task {tid} failed: {e}")
                traceback.print_exc()
                continue

            if solved and hyp:
                success_count += 1
                print(f"当前成功记录数: {success_count}")
                test_pairs = get_test_pairs(tid, train_tasks, train_sols)
                for t_idx, pair in enumerate(test_pairs):
                    test_dir = os.path.join(args.out, tid, f"test{t_idx}")
                    print_grid(pair["input"], f"Test {t_idx} input")
                    print_grid(pair["output"], f"Test {t_idx} expected")
                    bk_path = generate_test_bk(
                        pair["input"],
                        pair["output"],
                        test_dir,
                        enable_pi=args.enable_pi,
                        background_color=None,
                        pixel_threshold_pct=args.bg_threshold,
                    )
                    meta_path = os.path.join(test_dir, "grid_meta.json")
                    pred_grid = predict_from_prolog(hyp, bk_path, meta_path, pair_id="p0")
                    print_grid(pred_grid, f"Test {t_idx} predicted")
                    save_grid_txt(pred_grid, os.path.join(test_dir, "pred.txt"))
                    exact, pix_acc = evaluate_prediction(pred_grid, pair["output"])
                    print(f"Test {t_idx} - Exact match? {exact}")
                    print(f"Test {t_idx} - Pixel accuracy: {pix_acc}")

            print('\n')
            print(f"Finished {tid}\n")



if __name__ == "__main__":
    main()
