

def 

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

        # 如果发现尺寸不一致，跳过当前任务
        if skip_task:
            print(f"跳过任务 {tid} 处理，继续下一个任务")
            continue


        save_file = f"{FIGURES_PATH}/{tid}_train.png"
        print(f'Train task {jj}: {tid}')
        plot_task(task, f"  origin ARC grid show : ({jj}) {tid}   {train_or_eval}",
                  task_solution=task_solution,
                  save_file=None)
        # time.sleep(0.5)
        try:
            weight_grids = apply_object_weights_for_arc_task(task)

            # plot_weight_grids(weight_grids, f"权重网格 - 任务 {tid}")
            plot_weight_grids(weight_grids, f" ! Machine can see the gird weight ! - ({jj}) {tid}")

            # plot_weight_grids2(weight_grids, f" ! Machine can see the gird weight ! - ({jj}) {tid}")

            for grid_id, weight_grid in weight_grids.items():
            # grid_id 是如 'train_input_0', 'train_output_0', 'test_input_0' 这样的字符串
                display_weight_grid(weight_grid, title=f"{grid_id}")

        except Exception as e:
            print(f"无法绘制权重网格: {e}")

        # print("\n\nlen object_sets ",len(object_sets))
        print("\n\n\n\n")


        # if jj == 4: break


    except Exception as e:
        print(f"\n处理任务 {tid} 时出错: {str(e)}")
        print("继续处理下一个任务...\n")
        continue
