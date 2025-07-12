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

logger = logging.getLogger(__name__)
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


def save_task_counts(task_counts: List[Tuple[str, int]], path: str) -> None:
    """Save sorted ``task_counts`` list to ``path`` for offline analysis."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        for tid, cnt in task_counts:
            f.write(f"{tid} {cnt}\n")
    logger.debug("Saved task counts to %s", path)


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
    popper_timeout: int,
    max_clauses: int,
    max_vars: int,
    max_body: int,
    popper_debug: bool = True,
    show_stdout: bool = True,
) -> Tuple[bool, str | None]:
    """Generate Popper files for a task and run the solver.

    Parameters
    ----------
    popper_timeout : int
        Maximum time allowed for Popper in seconds.
    max_clauses, max_vars, max_body : int
        Bias parameters passed to :func:`generate_bias`.
    """
    out_dir = os.path.join(output_base, task_id)
    try:
        bg_color = determine_background_color(
            task_data, pixel_threshold_pct=bg_threshold, debug=True
        )
        if popper_debug :
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
            max_clauses=max_clauses,
            max_vars=max_vars,
            max_body=max_body,
        )
        # for label, path in ("BK", bk_path), ("Bias", bias_path), ("Examples", exs_path):
        #     print(f"\n============================================== {label} for {task_id} =======================")
        #     with open(path) as f:
        #         print(f.read())
        prog_path, info = run_popper_from_dir(
            out_dir,
            timeout=popper_timeout,
            debug=popper_debug,
            show_output=show_stdout,
        )
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


def run_test_evaluation(
    hyp_path: str,
    task_id: str,
    train_tasks: Dict[str, Any],
    train_sols: Dict[str, Any],
    *,
    output_base: str,
    enable_pi: bool,
    bg_threshold: int,
) -> List[Dict[str, Any]]:
    """Run prediction tests for ``task_id`` and collect metrics."""

    logger.debug("Running evaluation for %s", task_id)
    results = []
    pairs = get_test_pairs(task_id, train_tasks, train_sols)
    for t_idx, pair in enumerate(pairs):
        test_dir = os.path.join(output_base, task_id, f"test{t_idx}")
        print_grid(pair["input"], f"Test {t_idx} input")
        print_grid(pair["output"], f"Test {t_idx} expected")
        bk_path = generate_test_bk(
            pair["input"],
            pair["output"],
            test_dir,
            enable_pi=enable_pi,
            background_color=None,
            pixel_threshold_pct=bg_threshold,
        )
        meta_path = os.path.join(test_dir, "grid_meta.json")
        pred_grid = predict_from_prolog(hyp_path, bk_path, meta_path, pair_id="p0")
        print_grid(pred_grid, f"Test {t_idx} predicted")
        save_grid_txt(pred_grid, os.path.join(test_dir, "pred.txt"))
        logger.debug("Evaluation %s test %s predicted grid saved", task_id, t_idx)
        exact, pix_acc = evaluate_prediction(pred_grid, pair["output"])
        print(f"Test {t_idx} - Exact match? {exact}")
        print(f"Test {t_idx} - Pixel accuracy: {pix_acc}")
        results.append({"index": t_idx, "exact": exact, "pixel_accuracy": pix_acc})
        logger.debug(
            "Evaluation %s test %s result: exact=%s, acc=%s",
            task_id,
            t_idx,
            exact,
            pix_acc,
        )

    return results


def all_tests_successful(results: List[Dict[str, Any]]) -> bool:
    """Return ``True`` if all test cases are exact matches with 100% accuracy."""

    return all(r["exact"] and r["pixel_accuracy"] == 1.0 for r in results)


def print_progress(success_count: int, completed: int, total: int) -> None:
    """Display progress information for completed tasks."""

    fail_count = completed - success_count
    print(
        f"Progress - 成功 {success_count}/{total} 失败 {fail_count}/{total}"
    )



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
    parser.add_argument(
        "--popper-timeout",
        type=int,
        default=200,
        help="Maximum time allowed for Popper per task (seconds)",
    )
    parser.add_argument(
        "--max-clauses",
        type=int,
        default=4,
        help="Bias setting max_clauses",
    )
    parser.add_argument(
        "--max-vars",
        type=int,
        default=6,
        help="Bias setting max_vars",
    )
    parser.add_argument(
        "--max-body",
        type=int,
        default=4,
        help="Bias setting max_body",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.debug:
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

    sort_path = os.path.join(Path(__file__).resolve().parent, "analy", "task_counts.txt")
    save_task_counts(task_counts, sort_path)

    os.makedirs(args.out, exist_ok=True)

    success_count = 0
    all_results: Dict[str, List[Dict[str, Any]]] = {}
    tasks_completed = 0
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
            logger.debug("Queueing task %s", tid)
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
                popper_timeout=args.popper_timeout,
                max_clauses=args.max_clauses,
                max_vars=args.max_vars,
                max_body=args.max_body,
                # popper_debug=args.popper_debug,
                popper_debug=args.debug,
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
                tasks_completed += 1
                print_progress(success_count, tasks_completed, total_tasks)
                continue

            if solved and hyp:
                task_results = run_test_evaluation(
                    hyp,
                    tid,
                    train_tasks,
                    train_sols,
                    output_base=args.out,
                    enable_pi=args.enable_pi,
                    bg_threshold=args.bg_threshold,
                )

                all_results[tid] = task_results
                print(f"Task {tid} results: {task_results}")
                if all_tests_successful(task_results):
                    success_count += 1

            tasks_completed += 1
            print_progress(success_count, tasks_completed, total_tasks)

            print('\n')
            print(f"Finished {tid}\n")
            logger.debug("Completed task %s", tid)

    print("\n=== Summary ===")
    print(f"Total solved: {success_count}/{total_tasks}")
    for tid, results in all_results.items():
        print(f"Results for {tid}:")
        for res in results:
            print(f"  Test {res['index']} - exact={res['exact']} acc={res['pixel_accuracy']}")



if __name__ == "__main__":
    main()
