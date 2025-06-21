




# from
import traceback


from init.init import  prepare_arc_data

train_tasks, train_sols, eval_tasks, eval_sols, test_tasks = prepare_arc_data()



for jj, tid in enumerate(train_tasks):



    taskid = tid

    task = gettask()

    # 1. 抽取基础事实
    bias = None

    bk = None

    exs = None

    facts = extract_cell_facts(grids_canon)
    # 2. 分析 Δ-diff
    diff = diff_maps(grids_canon)
    facts += diff2facts(diff)
    # 3. 如果触发对象级 → segment & object_facts
    # 4. 生成 h_line/v_line 线索
    facts += detect_lines(diff)
    # 5. 写 bk.pl
    write_bk(facts, geometry_library)

    # 6. 写 exs.pl
    write_examples(grids_canon)

    # 7. Popper 搜索
    rules = popper_run(bk, bias, exs)

    # 8. 若 orientation ≠ 0°, 把规则输出结果逆旋转
























