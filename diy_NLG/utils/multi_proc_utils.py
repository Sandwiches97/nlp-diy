# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from multiprocessing import cpu_count, Pool

# cpu 数量
cores = cpu_count()
# 分块个数
partitions = cores

def paralleize(df, func):
    # data split
    data_split = np.array_split(df, partitions)
    # thread pool 线程过多会带来调度开销，进而影响缓存局部性和整体性能。
    # 而线程池维护着多个线程，等待着监督管理者分配可并发执行的任务。
    pool = Pool(cores)
    # 数据分发，合并
    data = pd.concat(pool.map(func, data_split))
    # 关闭线程池
    pool.close()
    # 执行完 close 后，不会又新的进程加入 pool
    # join 函数等待所有 子进程 结束
    pool.join()
    return data
