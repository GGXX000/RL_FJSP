"""
配置模块，包含所有全局配置参数
"""

class Config:
    # 图神经网络配置
    GNN_LAYERS = 3      # 图神经网络层数
    GNN_HIDDEN_SIZE = 1024  # 图神经网络隐藏层大小
    USE_ATTENTION = True  # 是否使用图注意力机制

    # 动作空间配置
    MAX_JOBS = 20    # 最大任务数，用于计算动作空间大小

    # DQN网络配置
    HIDDEN_SIZE = 2048  # 隐藏层大小，更改为与实际使用值一致

    # 学习参数
    BATCH_SIZE = 64
    GAMMA = 0.85      # 折扣因子
    EPSILON = 1.0      # 初始探索率
    EPSILON_MIN = 0.01 # 最小探索率
    EPSILON_DECAY = 0.998 # 探索率衰减

    # 图优化训练参数
    LEARNING_RATE = 0.003 # 学习率，降低学习率防止不稳定
    MEMORY_SIZE = 50000   # 经验回放缓冲区大小
    TARGET_UPDATE = 20    # 目标网络更新频率，增加频率

    # 正则化和训练稳定性
    WEIGHT_DECAY = 5e-5  # 权重衰减系数
    GRADIENT_CLIP = 1.0  # 梯度裁剪阈值
    DROPOUT_RATE = 0.2   # Dropout比率

    # 训练参数
    MAX_STEPS = 1000       # 每个回合的最大步数
    
    # 环境配置
    TOOL_CHANGE_TIME = 2  # 工具更换时间
    MAX_SLOTS = 7         # 每台机器的最大槽位数
    
    @classmethod
    def get_action_size(cls, num_machines):
        """计算动作空间大小"""
        return num_machines * cls.MAX_JOBS