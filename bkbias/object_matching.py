from __future__ import annotations
from typing import List, Dict, Any, Tuple


def _analyze_input_to_output_transformation(
    pair_id: int,
    input_grid: List[List[int]],
    output_grid: List[List[int]],
    input_obj_infos: List[Dict[str, Any]],
    output_obj_infos: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze how input objects transform into output objects.

    This is a simplified port of the old `_analyze_input_to_output_transformation`
    routine used in the previous project.  It works with the dictionary based
    object infos produced by :func:`extract_objects_from_task`.

    The function identifies preserved, modified, removed and added objects based
    on shape equality (using ``obj_000``) and tracks simple position and color
    changes.
    """
    transformation_rule: Dict[str, Any] = {
        "pair_id": pair_id,
        "preserved_objects": [],
        "modified_objects": [],
        "removed_objects": [],
        "added_objects": [],
    }

    unmatched_out = output_obj_infos.copy()

    for in_obj in input_obj_infos:
        match = None
        for out_obj in unmatched_out:
            if in_obj["obj_000"] == out_obj["obj_000"]:
                match = out_obj
                break
        if match is not None:
            delta_row = match["top"] - in_obj["top"]
            delta_col = match["left"] - in_obj["left"]
            color_change = (
                None
                if in_obj.get("main_color") == match.get("main_color")
                else {
                    "from": in_obj.get("main_color"),
                    "to": match.get("main_color"),
                }
            )
            if delta_row == 0 and delta_col == 0 and color_change is None:
                transformation_rule["preserved_objects"].append(
                    {
                        "input_obj_id": in_obj["obj_id"],
                        "output_obj_id": match["obj_id"],
                        "object": in_obj,
                    }
                )
            else:
                transformation_rule["modified_objects"].append(
                    {
                        "input_obj_id": in_obj["obj_id"],
                        "output_obj_id": match["obj_id"],
                        "position_change": {
                            "delta_row": delta_row,
                            "delta_col": delta_col,
                        },
                        "color_change": color_change,
                    }
                )
            unmatched_out.remove(match)
        else:
            transformation_rule["removed_objects"].append(
                {
                    "input_obj_id": in_obj["obj_id"],
                    "object": in_obj,
                }
            )

    for out_obj in unmatched_out:
        transformation_rule["added_objects"].append(
            {
                "output_obj_id": out_obj["obj_id"],
                "object": out_obj,
            }
        )

    return transformation_rule


def analyze_task_transformations(
    task_data: Dict[str, Any],
    all_objects: Dict[str, List[Tuple[int, List[Dict[str, Any]]]]],
) -> List[Tuple[int, Dict[str, Any]]]:
    """Analyze transformations for all train pairs in a task."""
    transformations: List[Tuple[int, Dict[str, Any]]] = []
    for (pair_id, in_objs), (_, out_objs) in zip(
        all_objects.get("input", []), all_objects.get("output", [])
    ):
        pair = task_data["train"][pair_id]
        rule = _analyze_input_to_output_transformation(
            pair_id,
            pair["input"],
            pair["output"],
            in_objs,
            out_objs,
        )
        transformations.append((pair_id, rule))
    return transformations

