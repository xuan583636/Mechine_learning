# -*- coding: utf-8 -*-
# @Time    : 2017/5/16 11:14
# @Author  : UNE
# @Project : Mechine_learning
# @File    : 邮件分类.py
# @Software: PyCharm

import re

ragEx = re.compile('\\w*')      # 除去单词，数字外的任意字符