import random
import os


def generate_job_shop_data(num_jobs, num_operation_types, min_operations=2, max_operations=8, min_time=2, max_time=20):
    """
    生成车间作业调度数据

    参数:
    num_jobs: 任务数量
    num_operation_types: 工序类型数量
    min_operations: 每个任务最少工序数
    max_operations: 每个任务最多工序数
    min_time: 最小处理时间
    max_time: 最大处理时间

    返回:
    data_string: 生成的数据字符串
    """
    # 第一行: 任务数量和工序类型数量
    data_string = f"{num_jobs} {num_operation_types}\n\n"

    # 为每个任务生成工序
    for job in range(num_jobs):
        # 决定这个任务有多少个工序
        num_operations = random.randint(min_operations, max_operations)

        # 生成这个任务的工序
        job_data = []
        # 为了确保工序类型不重复，我们先随机选取一些工序类型
        operation_types = random.sample(range(1, num_operation_types + 1), num_operations)

        for op_type in operation_types:
            # 工序类型和处理时间
            processing_time = random.randint(min_time, max_time)
            job_data.extend([op_type, processing_time])

        # 添加到数据字符串
        data_string += " ".join(map(str, job_data)) + "\n"

    return data_string


def generate_multiple_datasets(num_datasets, output_dir="job_shop_data",
                               num_jobs_range=(50, 100),
                               num_operation_types_range=(10, 30),
                               min_operations=2,
                               max_operations=8,
                               min_time=2,
                               max_time=20):
    """
    生成多个数据集文件

    参数:
    num_datasets: 要生成的数据集数量
    output_dir: 输出目录
    num_jobs_range: 任务数量范围 (min, max)
    num_operation_types_range: 工序类型数量范围 (min, max)
    min_operations: 每个任务最少工序数
    max_operations: 每个任务最多工序数
    min_time: 最小处理时间
    max_time: 最大处理时间
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 生成数据集
    for i in range(num_datasets):
        num_jobs = random.randint(*num_jobs_range)
        num_operation_types = random.randint(*num_operation_types_range)

        data_string = generate_job_shop_data(
            num_jobs,
            num_operation_types,
            min_operations,
            max_operations,
            min_time,
            max_time
        )

        # 写入文件
        file_path = os.path.join(output_dir, f"data_{i + 1}.txt")
        with open(file_path, 'w') as f:
            f.write(data_string)

        print(f"生成数据集 {i + 1}/{num_datasets}: {num_jobs}个任务, {num_operation_types}种工序类型")


if __name__ == "__main__":
    # 创建一个目录来存放生成的数据集
    output_dir = "datasets"

    # 生成100个数据集文件，每个数据集至少50个任务
    generate_multiple_datasets(
        num_datasets=2000,
        output_dir=output_dir,
        num_jobs_range=(10, 20),  # 任务数量在50到100之间随机
        num_operation_types_range=(15, 20),  # 工序类型数量在15到30之间随机
        min_operations=4,  # 每个任务至少2个工序
        max_operations=6,  # 每个任务最多6个工序
        min_time=2,  # 最小处理时间2
        max_time=15  # 最大处理时间15
    )

    print(f"\n成功生成100个数据集，保存在 {output_dir} 目录下")
    print("每个数据集包含50-100个任务，可用于训练强化学习模型")