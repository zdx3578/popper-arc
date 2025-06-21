import json
import os
from typing import Dict, Any, List, Optional
import sys
import logging
import traceback
import cgitb
import time
import subprocess

# 导入增强型差异分析器
# from diffstar_weighted_arc_analyzer import WeightedARCDiffAnalyzer
# from weighted_analyzer import WeightedARCDiffAnalyzer
from arcMrule.diffstar.weighted_analyzer import WeightedARCDiffAnalyzer

# 启用CGI追踪以便更好的错误报告
cgitb.enable(format='text')

class WeightedARCSolver:
    """ARC问题解决器，使用带权重的差异网格分析"""

    def __init__(self, data_dir=None, debug=True, pixel_threshold_pct=40):
        """
        初始化增强型ARC解决器

        Args:
            data_dir: ARC数据集目录，如果提供则自动加载整个数据集
            debug: 是否开启调试模式
            pixel_threshold_pct: 颜色占比阈值（百分比），超过此阈值的颜色视为背景
        """
        # 创建加权差异分析器
        self.diff_analyzer = WeightedARCDiffAnalyzer(
            debug=debug,
            debug_dir="weighted_debug_output",
            pixel_threshold_pct=pixel_threshold_pct,  # 背景颜色阈值
            weight_increment=2,      # 对象权重增量
            diff_weight_increment=5  # 差异区域权重增量
        )

        self.debug = debug

        # 初始化数据字典
        self.train_tasks = {}
        self.train_sols = {}
        self.eval_tasks = {}
        self.eval_sols = {}
        self.test_tasks = {}

        # 如果提供了数据目录，则加载整个数据集
        if data_dir:
            self.data_dir = data_dir
            self.load_all_data()

    def find_data_dir(self):
        """查找可用的数据目录路径"""
        # 定义多个可能的路径
        code_paths = [
            '/kaggle/input/arc-prize-2025',
            '/Users/zhangdexiang/github/VSAHDC/arcMrule',
            '/Users/zhangdexiang/github/VSAHDC/arcv2',
            '/home/zdx/github/VSAHDC/arcv2',
            "/home/zdx/github/VSAHDC/arcMrule",
        ]
        for path in code_paths:
            if os.path.exists(path):
                try:
                    if self.debug:
                        print(f"正在检查代码仓库: {path}")

                    # 使用subprocess代替os.system来捕获输出
                    result = subprocess.run(
                        f'cd {path} && git pull',
                        shell=True,
                        capture_output=True,
                        text=True
                    )

                    # 分析输出结果判断是否有更新
                    output = result.stdout.strip()
                    if self.debug:
                        print(f"Git更新结果: {output}")

                    # 如果不包含"Already up to date."或中文"已经是最新"，说明代码被更新了
                    if "Already up to date." not in output and "已经是最新" not in output and output:
                        print(f"代码仓库 {path} 已更新，程序将退出以加载新代码")
                        sys.exit(0)  # 退出程序，返回状态码0表示正常退出

                except Exception as e:
                    print(f"检查代码仓库更新失败: {path}, 错误: {e}")

        time.sleep(0.001)

        data_paths = [
            '/kaggle/input/arc-prize-2025',
            '/Users/zhangdexiang/github/ARC-AGI-2/arc-prize-2025',
            # '/home/zdx/github/VSAHDC/arc-prize-2025',
            "/home/zdx/github/ARC-AGI-2/arc-prize-2025/",
            './data',
            './arc-data'
        ]

        # 遍历路径列表，找到第一个存在的路径
        for path in data_paths:
            if os.path.exists(path):
                self.data_dir = path
                if self.debug:
                    print(f"数据路径设置为: {self.data_dir}")
                print(f"数据路径设置为: {self.data_dir}")
                return path

        # 如果所有路径都不存在，抛出异常
        raise FileNotFoundError("未找到任何指定的数据路径！")

    def load_all_data(self):
        """加载所有ARC数据集文件"""
        if not hasattr(self, 'data_dir') or not self.data_dir:
            self.find_data_dir()

        # 尝试加载所有数据文件
        try:
            # 训练任务和解决方案
            train_tasks_path = os.path.join(self.data_dir, 'arc-agi_training_challenges.json')
            train_sols_path = os.path.join(self.data_dir, 'arc-agi_training_solutions.json')
            if os.path.exists(train_tasks_path):
                self.train_tasks = self.load_json(train_tasks_path)
            if os.path.exists(train_sols_path):
                self.train_sols = self.load_json(train_sols_path)

            # 评估任务和解决方案
            eval_tasks_path = os.path.join(self.data_dir, 'arc-agi_evaluation_challenges.json')
            eval_sols_path = os.path.join(self.data_dir, 'arc-agi_evaluation_solutions.json')
            if os.path.exists(eval_tasks_path):
                self.eval_tasks = self.load_json(eval_tasks_path)
            if os.path.exists(eval_sols_path):
                self.eval_sols = self.load_json(eval_sols_path)

            # 测试任务
            test_tasks_path = os.path.join(self.data_dir, 'arc-agi_test_challenges.json')
            if os.path.exists(test_tasks_path):
                self.test_tasks = self.load_json(test_tasks_path)

            if self.debug:
                print(f"加载了 {len(self.train_tasks)} 个训练任务")
                print(f"加载了 {len(self.eval_tasks)} 个评估任务")
                print(f"加载了 {len(self.test_tasks)} 个测试任务")

            return True

        except Exception as e:
            if self.debug:
                print(f"加载数据集失败: {e}")
            return False

    def load_json(self, file_path: str) -> Dict:
        """加载JSON文件"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            if self.debug:
                print(f"加载JSON文件失败: {file_path}")
                print(f"错误: {str(e)}")
            return {}

    def load_task(self, task_id_or_json: str) -> Optional[Dict[str, Any]]:
        """
        加载ARC任务，可以是任务ID或JSON文件路径

        Args:
            task_id_or_json: 任务ID或JSON文件路径

        Returns:
            加载的任务数据
        """
        # 判断是任务ID还是文件路径
        if os.path.exists(task_id_or_json) and task_id_or_json.endswith('.json'):
            # 加载单个JSON文件
            try:
                with open(task_id_or_json, 'r') as f:
                    task_data = json.load(f)

                # 验证任务格式
                if 'train' not in task_data or 'test' not in task_data:
                    raise ValueError("任务JSON格式不正确，缺少train或test字段")

                return {'task': task_data, 'id': os.path.basename(task_id_or_json).replace('.json', '')}
            except Exception as e:
                print(f"加载任务文件失败: {e}")
                return None
        else:
            # 作为任务ID处理
            task_id = task_id_or_json

            # 如果未加载数据集，先加载
            if not hasattr(self, 'train_tasks') or not self.train_tasks:
                self.load_all_data()

            # 在各个数据集中查找任务
            if task_id in self.train_tasks:
                return {
                    'task': self.train_tasks[task_id],
                    'solution': self.train_sols.get(task_id),
                    'id': task_id,
                    'type': 'train'
                }
            elif task_id in self.eval_tasks:
                return {
                    'task': self.eval_tasks[task_id],
                    'solution': self.eval_sols.get(task_id),
                    'id': task_id,
                    'type': 'eval'
                }
            elif task_id in self.test_tasks:
                return {
                    'task': self.test_tasks[task_id],
                    'solution': None,  # 测试任务没有解决方案
                    'id': task_id,
                    'type': 'test'
                }
            else:
                print(f"未找到任务ID: {task_id}")
                return None

    def determine_background_color(self, task_data):
        """
        分析所有训练数据确定背景色，只返回占比最大的背景色值

        Args:
            task_data: 任务数据

        Returns:
            背景色值(单个整数)，如果没有满足条件的背景色则返回None
        """
        from collections import defaultdict

        all_grids = []

        # 收集所有输入和输出网格
        for example in task_data['train']:
            all_grids.append(example['input'])
            all_grids.append(example['output'])

        # 统计所有网格中各颜色的总占比
        color_total_percentages = defaultdict(float)
        color_appearance_count = defaultdict(int)

        for grid in all_grids:
            if not grid:  # 跳过空网格
                continue

            total_pixels = len(grid) * len(grid[0])
            if total_pixels == 0:  # 避免除零错误
                continue

            color_counts = defaultdict(int)

            for row in grid:
                for cell in row:
                    color_counts[cell] += 1

            # 计算每种颜色的百分比并累加
            for color, count in color_counts.items():
                percentage = (count / total_pixels * 100)
                color_total_percentages[color] += percentage
                color_appearance_count[color] += 1

        # 计算每种颜色的平均占比
        color_avg_percentages = {}
        for color, total_pct in color_total_percentages.items():
            color_avg_percentages[color] = total_pct / color_appearance_count[color]

        # 按平均占比排序颜色
        sorted_colors = sorted(
            color_avg_percentages.items(),
            key=lambda x: x[1],
            reverse=True
        )

        background_color = None

        # 只获取占比最大的颜色作为背景色
        if sorted_colors:
            max_color, max_percentage = sorted_colors[0]
            # 确保占比超过阈值
            if max_percentage >= self.diff_analyzer.pixel_threshold_pct:
                background_color = max_color
                if self.debug:
                    print(f"确定全局背景色: {max_color} (占比: {max_percentage:.2f}%)")

        return background_color


    def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理ARC任务数据，使用加权分析器

        Args:
            task_data: 任务数据

        Returns:
            处理结果
        """
        # 获取实际任务数据
        task = task_data['task']
        task_id = task_data.get('id', 'unknown')

        if self.debug:
            print(f"\n\n\n\n\n\n处理 task 任务 {task_id}")

        # 重置差异分析器
        self.diff_analyzer = WeightedARCDiffAnalyzer(
            debug=self.debug,
            debug_dir=f"weighted_debug_output/{task_id}"
        )

        # 创建输出目录
        if self.debug:
            os.makedirs(f"weighted_debug_output/{task_id}", exist_ok=True)

        # 处理训练数据
        param_combinations3: List[Tuple[bool, bool, bool]] = [
            (False, False, False),
            (False, False, True),
            (False, True, False),
            (False, True, True),
            (True, True, False),
            (True, True, True),
            (True, False, False),
            (True, False, True)  ]

        param_combinations2: List[Tuple[bool, bool, bool]] = [
            (True, True, False),
            (True, True, True),
            (True, False, False),
            (True, False, True),
            (False, False, False),
            (False, False, True),
            (False, True, False),
            (False, True, True) ]

        param_combinations = [
            (True, True, False),
            (True, False, False),
            (False, False, False),
            (False, True, False)         ]
        param_combinations0 = [
            (True, True, False),
            (True, False, False),
            (False, False, False),
            (False, True, False)       ]

        # 在处理训练数据前确定背景色
        background_color = self.determine_background_color(task)
        # background_color = set(background_color)  # 转换为集合以去重
        self.diff_analyzer.set_background_colors(background_color)

        if self.debug and background_color:
            print(f"任务 {task_id} 的全局背景色: {background_color}")

        for param in param_combinations:
            print(f"\n\n\n\n\n\n处理任务 {task_id} 的参数组合: {param}")
            for i, example in enumerate(task['train']):
                print(f"\n\n\n\n处理 task TRAIN pair 数据对 {i} \n\n")
                input_grid = example['input']
                output_grid = example['output']

                # 添加到差异分析器
                #! when again param loop add obj to one diff_analyzer for  6: 025d127b
                self.diff_analyzer.add_train_pair(i, input_grid, output_grid, param, background_color)
            for i, example in enumerate(task['test']):
                print(f"\n\n\n\n处理 task TEST pair 数据对 {i} \n\n")
                input_grid = example['input']
                if 'output' not in example or example['output'] is None:
                    output_grid = None
                else:
                    output_grid = example['output']

                # 添加到差异分析器
                self.diff_analyzer.add_test_pair(i, input_grid, output_grid, param, background_color,test=True)

            common_patterns = self.diff_analyzer.enhanced_analyze_common_patterns_with_test_data_matching(task)






            # 处理测试数据
            test_predictions = []
            test_success_count = 0
            total_test_count = len(task['test'])

            # 先在训练数据上验证规则的有效性
            train_validation_results = []
            train_success_count = 0
            print(f"\n\n\n\n\n\n处理训练数据验证")


            # 记录训练数据验证的成功率
            train_success_rate = 1 #train_success_count / len(task['train']) if task['train'] else 0

            # 现在处理测试数据
            if self.debug:
                print(f"\n\n\n\n\n===== 处理测试数据 =====")
                # print(f"\n\n\n\n\n训练验证成功率: {train_success_rate:.2f}\n\n\n\n\n")
                print(f"测试样例数量: {total_test_count}\n\n\n\n\n")

            for i, example in enumerate(task['test']):
                if self.debug:
                    print(f"\n----- 测试样例 {i+1}/{total_test_count} -----")

                input_grid = example['input']

                if self.debug:
                    print(f"输入网格尺寸: {len(input_grid)}x{len(input_grid[0])}")

                # 1. 使用权重分析模式进行预测 #! old code pass
                weight_based_prediction = None #self.diff_analyzer.apply_common_patterns(input_grid, param)

                # 2. 使用转换规则进行预测
                transform_based_prediction = self.diff_analyzer.apply_transformation_rules( 'test', i,
                    input_grid,
                    common_patterns,                )

                if self.debug:
                    # print(f"权重模式预测网格尺寸: {len(weight_based_prediction)}x{len(weight_based_prediction[0]) if weight_based_prediction else 0}")
                    print(f"转换规则预测网格尺寸: {len(transform_based_prediction)}x{len(transform_based_prediction[0]) if transform_based_prediction else 0}")

                # 3. 确定最终预测结果（优先使用训练验证成功率更高的方法）
                final_prediction = None
                prediction_method = ""
                prediction_confidence = 0.0

                if train_success_rate > 0.5:  # 如果转换规则在训练数据上的成功率超过50%
                    final_prediction = transform_based_prediction
                    prediction_method = "transformation_rules"
                    # 计算预测置信度
                    prediction_confidence = self.diff_analyzer.calculate_rule_confidence(
                        input_grid, transform_based_prediction
                    )

                else:
                    final_prediction = weight_based_prediction
                    prediction_method = "weight_patterns"

                    if self.debug:
                        print(f"使用方法: 权重模式 (训练成功率 <= 0.5)")

                if self.debug:
                    print(f"预测方法: {prediction_method}")

                if True:
                    print(f"\n\n\n\n\n\n处理任务 {task_id} 的参数组合: {param}")

                if 'solution' in task_data and task_data['solution']:
                    actual_output = task_data['solution'][i]
                    #!
                    is_correct = final_prediction == actual_output
                    if is_correct:
                        test_success_count += 1
                        if self.debug:
                            print(f"预测结果: 正确 ✓")
                    else:
                        if self.debug:
                            print(f"预测结果: 错误 ✗")
                    if test_success_count == total_test_count:
                        if self.debug:
                            print(f"\n===== 所有测试用例验证成功！提前结束 =====")
                        return {
                            'task_id': task_id,
                            'common_patterns': common_patterns,
                            'test_success': True,
                            'test_success_count': test_success_count,
                            'total_test_count': total_test_count,
                            'success_rate': 1.0,
                            'param_combination': param
                        }

                else:
                    # 没有解决方案，只返回预测
                    if self.debug:
                        print(f"预测结果: 未知（无参考解决方案）")

            # 输出总体测试结果
            if self.debug:
                if total_test_count > 0 and 'solution' in task_data and task_data['solution']:
                    success_rate = test_success_count / total_test_count
                    print(f"\n===== 测试结果汇总 =====")
                    print(f"任务ID: {task_id}")
                    print(f"总测试样例: {total_test_count}")
                    print(f"正确预测: {test_success_count}")
                    print(f"正确率: {success_rate:.2f} ({test_success_count}/{total_test_count})")
                    print(f"使用参数: {param}")
                else:
                    print(f"\n===== 预测完成 =====")
                    print(f"任务ID: {task_id}")
                    print(f"总测试样例: {total_test_count}")
                    print(f"使用参数: {param}")


    def process_all_tasks(self, task_type='train', limit=None):
        """
        处理指定类型的所有任务

        Args:
            task_type: 任务类型，可以是'train', 'eval'或'test'
            limit: 处理的最大任务数，None表示处理所有任务

        Returns:
            所有任务的处理结果
        """
        # 确保数据已加载
        if not hasattr(self, 'train_tasks') or not self.train_tasks:
            self.load_all_data()

        # 选择任务集
        if task_type == 'train':
            task_dict = self.train_tasks
            solutions = self.train_sols
        elif task_type == 'eval':
            task_dict = self.eval_tasks
            solutions = self.eval_sols
        elif task_type == 'test':
            task_dict = self.test_tasks
            solutions = {}
        else:
            raise ValueError(f"未知的任务类型: {task_type}")

        # 处理所有任务
        results = {}
        count = 0
        success_count = 0

        for task_id, task in task_dict.items():
            if limit is not None and count >= limit:
                break

            # if task_id != '00d62c1b':
            if task_id != '009d5c81':
                continue

            task_data = {
                'task': task,
                'solution': solutions.get(task_id),
                'id': task_id,
                'type': task_type
            }


            # if task_id =='00d62c1b':
            #     print(f"\n\n\n\n处理任务 {task_id} 的参数组合: {task_data['solution']}")


            try:
                result = self.process_task(task_data)
                results[task_id] = result
                count += 1

                # 检查任务是否成功解决
                if result and result.get('test_success', False):
                    success_count += 1
                    if True: #self.debug:
                        print(f"任务 {task_id} 完全成功解决! ({success_count}/{count})")

                if True: #self.debug:
                    print(f"完成处理任务 {task_id} ({count}/{len(task_dict) if limit is None else min(limit, len(task_dict))})")
                    if success_count > 0:
                        print(f"当前成功率: {success_count/count:.2f} ({success_count}/{count})")

            except Exception as e:
                if self.debug:
                    print(f"处理任务 {task_id} 失败: {e}")
                tb_str = traceback.format_exc()
                print(f"发生错误: {e}")
                print(f"错误上下文:\n{tb_str}")
                exc_type, exc_value, exc_traceback = sys.exc_info()
                # 获取出错的文件名、行号、函数名和代码行
                filename, line_number, func_name, code_line = traceback.extract_tb(exc_traceback)[-1]
                print(f"错误: {e}")
                print(f"发生在文件 {filename}, 第 {line_number} 行, 函数 {func_name}")
                print(f"错误代码: {code_line}")

        # 返回结果和统计信息
        return {
            'results': results,
            'total_processed': count,
            'success_count': success_count,
            'success_rate': success_count/count if count > 0 else 0
        }

    def save_results(self, results: Dict[str, Any], output_file: str) -> bool:
        """
        保存处理结果

        Args:
            results: 处理结果
            output_file: 输出文件路径

        Returns:
            是否成功保存
        """
        try:
            # 处理结果以便JSON序列化
            serializable_results = self._make_serializable(results)

            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"结果已保存到 {output_file}")
            return True
        except Exception as e:
            print(f"保存结果失败: {e}")
            return False

    def _make_serializable(self, obj):
        """使对象可JSON序列化"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(self._make_serializable(item) for item in obj)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, frozenset):
            return list(self._make_serializable(item) for item in obj)
        else:
            # 其他类型转为字符串
            return str(obj)