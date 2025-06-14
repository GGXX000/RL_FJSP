import os
import pandas as pd
import openpyxl
from openpyxl.utils.exceptions import InvalidFileException


class ValidationLogger:
    """
    用于记录强化学习训练过程中验证集结果的模块
    记录格式: 每行表示一次验证，每列表示一个数据集
    单元格内容格式: [完工时间, 模具更换次数]
    """

    def __init__(self, log_file_path='validation_results.xlsx'):
        """
        初始化验证结果记录器

        参数:
            log_file_path: Excel文件路径，默认为'validation_results.xlsx'
        """
        self.log_file_path = log_file_path
        self.validation_count = 0
        self.dataset_columns = set()

        # 如果文件存在，加载现有数据
        if os.path.exists(log_file_path):
            try:
                self.df = pd.read_excel(log_file_path, index_col=0)
                self.validation_count = len(self.df)
                self.dataset_columns = set(self.df.columns)
            except (InvalidFileException, Exception) as e:
                print(f"无法加载现有日志文件: {e}")
                self.df = pd.DataFrame()
        else:
            self.df = pd.DataFrame()

    def log_validation_result(self, dataset_name, completion_time, tool_change_count):
        """
        记录单个数据集的验证结果

        参数:
            dataset_name: 数据集名称
            completion_time: 完工时间
            tool_change_count: 模具更换次数
        """
        # 添加新的数据集列（如果尚不存在）
        if dataset_name not in self.dataset_columns:
            self.dataset_columns.add(dataset_name)
            if dataset_name not in self.df.columns:
                self.df[dataset_name] = None

        # 检查是否需要添加新的验证行
        validation_idx = f"验证_{self.validation_count + 1}"
        if validation_idx not in self.df.index:
            self.df.loc[validation_idx] = None

        # 更新结果
        self.df.at[validation_idx, dataset_name] = f"[{completion_time}, {tool_change_count}]"

    def complete_validation_round(self):
        """
        完成一轮验证，保存结果并更新验证计数
        """
        # 保存Excel文件
        self.save()

        # 更新验证计数
        self.validation_count += 1

        print(f"已完成第 {self.validation_count} 轮验证，结果已保存到 {self.log_file_path}")

    def save(self):
        """
        保存结果到Excel文件
        """
        try:
            # 保存DataFrame到Excel
            self.df.to_excel(self.log_file_path)
            return True
        except Exception as e:
            print(f"保存验证结果时出错: {e}")
            return False