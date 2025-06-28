import argparse
import os
import time
import traceback
from typing import Any, Dict, List, Tuple

from init.init import prepare_arc_data
from bkbias.objattr import (
    determine_background_color,
    generate_files_from_task,
    nonbg_pixels,
    run_popper_from_dir,
)


# Enable predicate invention by default
DEFAULT_ENABLE_PI = True


def count_non_background_pixels(task_data: Dict[str, Any]) -> int:
    """Return the total number of non-background pixels in all train grids."""
    bg_color = determine_background_color(task_data, debug=False)
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
) -> None:
    """Generate Popper files for a task and run the solver."""
    out_dir = os.path.join(output_base, task_id)
    try:
        bk_path, bias_path, exs_path = generate_files_from_task(
            task_data,
            out_dir,
            use_pixels=use_pixels,
            bk_use_pixels=bk_use_pixels,
            exs_use_pixels=exs_use_pixels,
            enable_pi=enable_pi,
        )
        for label, path in ("BK", bk_path), ("Bias", bias_path), ("Examples", exs_path):
            print(f"\n============================================== {label} for {task_id} =======================")
            with open(path) as f:
                print(f.read())
        prog, score, _ = run_popper_from_dir(out_dir)
        if prog is not None:
            print(f"！！！！！！！！！！！！！！！！！！！！！！Solved {task_id} with score {score}")
        else:
            print(f"No solution for {task_id}")
    except Exception as e:  # pragma: no cover - runtime errors
        print(f"Error processing {task_id}: {e}")
        traceback.print_exc()


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
    args = parser.parse_args()

    use_pixels = args.repr == "pixels"
    bk_use_pixels = None if args.bk_repr is None else args.bk_repr == "pixels"
    exs_use_pixels = None if args.exs_repr is None else args.exs_repr == "pixels"

    train_tasks, train_sols, eval_tasks, eval_sols, test_tasks = prepare_arc_data()
    task_counts: List[Tuple[str, int]] = []
    for tid, tdata in train_tasks.items():
        cnt = count_non_background_pixels(tdata)
        task_counts.append((tid, cnt))
    task_counts.sort(key=lambda x: x[1])

    os.makedirs(args.out, exist_ok=True)

    for tid, _ in task_counts:
        print(f"Processing {tid}")
        solve_task(
            tid,
            train_tasks[tid],
            output_base=args.out,
            use_pixels=use_pixels,
            bk_use_pixels=bk_use_pixels,
            exs_use_pixels=exs_use_pixels,
            enable_pi=args.enable_pi,
        )
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
