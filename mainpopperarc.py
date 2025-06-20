




# from
import traceback


from init import  prepare_arc_data

train_tasks, train_sols, eval_tasks, eval_sols, test_tasks = prepare_arc_data()



for jj, tid in enumerate(train_tasks):
    try:
        # tid = '009d5c81'
        if tid in train_tasks.keys():
            train_or_eval = 'train'
            task = train_tasks[tid]
            task_solution = train_sols[tid]
        else:
            train_or_eval = 'eval'
            task = eval_tasks[tid]
            task_solution = eval_sols[tid]
        # 检查所有训练样例的尺寸一致性
        skip_task = False
        train_data = task['train']
        for pair_id, data_pair in enumerate(train_data):
            I = data_pair['input']
            O = data_pair['output']

            # 获取输入和输出的尺寸
            height_i, width_i = len(I), len(I[0]) if I else 0
            height_o, width_o = len(O), len(O[0]) if O else 0

            # 检查尺寸是否一致
            if height_i != height_o or width_i != width_o:
                print(f"任务 {tid} 的样例 {pair_id} 尺寸不一致: 输入 {height_i}x{width_i}, 输出 {height_o}x{width_o}")
                skip_task = True
                break
    except Exception as e:
        print(f"Error processing task {tid}: {str(e)}")
        traceback.print_exc()

for each task:
    grids = load_json(...)
    orientation = argmin_entropy_rotation(grids)
    grids_canon = rotate_all(grids, orientation)

    # 1. 抽取基础事实
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
























