# -*- coding: utf-8 -*-
import re
import jieba
import pandas as pd
import numpy as np
import os
import sys

# 设置项目的 root 目录，方便后续相关代码package的导入
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

# 导入文本预处理的配置信息 config1
from config1 import *
# 导入多核CPU并行处理数据
from multi_proc_utils import *

# jieba 载入自定义切词表
jieba.load_userdict(user_dict_path)
