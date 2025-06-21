import os
import subprocess
import tempfile
from typing import List, Dict, Any, Tuple

class PopperIntegrator:
    """与Popper集成的工具类"""

    @staticmethod
    def prepare_popper_files(facts: List[str], positive: List[str], negative: List[str], bias: str, output_dir: str = None) -> str:
        """准备Popper输入文件"""
        # 如果未指定输出目录，创建临时目录
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="popper_")

        # 确保目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 写入文件
        with open(os.path.join(output_dir, "background.pl"), "w") as f:
            f.write("\n".join(facts))

        with open(os.path.join(output_dir, "positive.pl"), "w") as f:
            f.write("\n".join(positive))

        with open(os.path.join(output_dir, "negative.pl"), "w") as f:
            f.write("\n".join(negative))

        with open(os.path.join(output_dir, "bias.pl"), "w") as f:
            f.write(bias)

        return output_dir

    @staticmethod
    def run_popper(output_dir: str, popper_path: str = None) -> List[str]:
        """运行Popper并返回学习到的规则"""
        try:
            # 检查是否有Python API
            try:
                from popper.util import Settings
                from popper.loop import learn_solution

                settings = Settings(
                    bias_file=os.path.join(output_dir, "bias.pl"),
                    pos_file=os.path.join(output_dir, "positive.pl"),
                    neg_file=os.path.join(output_dir, "negative.pl"),
                    bk_file=os.path.join(output_dir, "background.pl"),
                    timeout=60
                )

                learned_rules = learn_solution(settings)
                return [str(rule) for rule in learned_rules] if learned_rules else []
            except ImportError:
                # 如果无法导入Popper Python API，尝试命令行
                if popper_path is None:
                    popper_path = "popper"  # 默认命令

                cmd = [
                    popper_path,
                    "--bias", os.path.join(output_dir, "bias.pl"),
                    "--pos", os.path.join(output_dir, "positive.pl"),
                    "--neg", os.path.join(output_dir, "negative.pl"),
                    "--bk", os.path.join(output_dir, "background.pl")
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"Popper运行失败: {result.stderr}")
                    return []

                # 解析输出以提取规则
                rules = []
                lines = result.stdout.split("\n")
                for line in lines:
                    if line.strip().startswith("extends_to_grid") or \
                       line.strip().startswith("yellow_fills_vertical") or \
                       line.strip().startswith("green_at_intersections"):
                        rules.append(line.strip())

                return rules

        except Exception as e:
            print(f"运行Popper时出错: {e}")
            return []

    @staticmethod
    def parse_rules(rules: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """解析Popper规则到可执行格式"""
        parsed_rules = {}

        for rule in rules:
            # 分离头部和体部
            if ":-" in rule:
                head, body = rule.split(":-", 1)
            else:
                head, body = rule, ""

            head = head.strip()
            body = body.strip()

            # 提取谓词名称
            pred_name = head.split("(")[0]

            # 解析条件
            conditions = []
            if body:
                for cond in body.split(","):
                    cond = cond.strip()
                    if cond:
                        # 解析条件谓词和参数
                        pred = cond.split("(")[0]
                        params_str = cond.split("(")[1].split(")")[0]
                        params = [p.strip() for p in params_str.split(",")]

                        conditions.append({
                            "predicate": pred,
                            "parameters": params
                        })

            # 将规则添加到相应的谓词
            if pred_name not in parsed_rules:
                parsed_rules[pred_name] = []

            parsed_rules[pred_name].append({
                "head": head,
                "conditions": conditions
            })

        return parsed_rules

    @staticmethod
    def extract_oneInOut_mapping_rules(input_objects: List[Dict], output_objects: List[Dict]) -> Dict:
        """从对象列表中提取映射规则"""
        # 实际实现会根据对象分析提取规则
        mapping_rule = {
            'preserved_objects': [],
            'modified_objects': [],
            'removed_objects': [],
            'added_objects': []
        }

        # 分析对象间映射关系
        # ...

        return mapping_rule

    @staticmethod
    def find_patterns_across_pairs(oneInOut_mapping_rules: Dict) -> List[Dict]:
        """查找跨数据对的模式"""
        # 实际实现会分析不同数据对之间的共同模式
        patterns = []

        # 查找模式逻辑
        # ...

        return patterns

    @staticmethod
    def extract_global_operation_rules(mapping_rules: Dict, patterns: List[Dict]) -> List[Dict]:
        """提取全局操作规则"""
        # 实际实现会从映射规则和模式中提取全局规则
        global_rules = []

        # 提取全局规则逻辑
        # ...

        return global_rules

    @staticmethod
    def extract_conditional_rules(mapping_rules: Dict, patterns: List[Dict]) -> List[Dict]:
        """提取条件规则"""
        # 实际实现会从映射规则和模式中提取条件规则
        conditional_rules = []

        # 提取条件规则逻辑
        # ...

        return conditional_rules

    @staticmethod
    def integrate_rules(global_rules: List[Dict], conditional_rules: List[Dict]) -> Dict:
        """整合全局规则和条件规则"""
        # 实际实现会整合不同类型的规则
        integrated_rules = {
            'global_rules': global_rules,
            'conditional_rules': conditional_rules
        }

        # 整合规则逻辑
        # ...

        return integrated_rules