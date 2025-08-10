import argparse
import os
from pathlib import Path
import time
import traceback
from typing import Any, Dict, List, Tuple

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
from bkbias.hyp_export import dump_hypothesis


# Enable predicate invention by default
DEFAULT_ENABLE_PI = True


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
        prog, score, _ = run_popper_from_dir(out_dir)
        if prog is not None:
            print(f"！！！！！！！！！！！！！！！！！！！！！！Solved {task_id} with score {score}")
            hyp_path = os.path.join(out_dir, "hyp.pl")
            if isinstance(prog, str):
                with open(hyp_path, "w") as f:
                    f.write(prog.strip() + "\n")
            else:
                dump_hypothesis(prog, Path(hyp_path))
            return True, hyp_path
        else:
            print(f"No solution for {task_id}")
            return False, None
    except Exception as e:  # pragma: no cover - runtime errors
        print(f"Error processing {task_id}: {e}")
        traceback.print_exc()
        return False, None


def main() -> None:
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
        default="0a2355a6",
        help="Run only the specified ARC task id",
    )
    args = parser.parse_args()

    use_pixels = args.repr == "pixels"
    bk_use_pixels = None if args.bk_repr is None else args.bk_repr == "pixels"
    exs_use_pixels = None if args.exs_repr is None else args.exs_repr == "pixels"

    train_tasks, train_sols, eval_tasks, eval_sols, test_tasks = prepare_arc_data()
    evaltesttask = False
    if evaltesttask:
        train_tasks, train_sols =   eval_tasks, eval_sols

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
    for idx, (tid, _) in enumerate(selected, start=1):
        print(f" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")

        print(f"Processing {tid} ({idx}/{total_tasks})")
        solved, hyp = solve_task(
            tid,
            train_tasks[tid],
            output_base=args.out,
            use_pixels=use_pixels,
            bk_use_pixels=bk_use_pixels,
            exs_use_pixels=exs_use_pixels,
            enable_pi=args.enable_pi,
            bg_threshold=args.bg_threshold,
        )
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
        try:
            # 等待用户敲 ↵ 或输入任意字符
            # input(f"\n[Epoch {epoch}] 按 Enter 继续，Ctrl-C 终止…")
            time.sleep(1)

        except KeyboardInterrupt:
            print("\n检测到用户中断，安全退出。")
            break
        print('\n')
        print('\n')
        print(f"Finished {tid}\n")
        print('\n')
        print('\n')
        print('\n')
        print('\n')
        print('\n')
        print('\n')



if __name__ == "__main__":
    main()
