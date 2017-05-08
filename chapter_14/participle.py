# -*- coding: utf-8 -*-
# @Time    : 2017/5/8 10:41
# @Author  : UNE
# @Project : Mechine_learning
# @File    : participle.py
# @Software: PyCharm

import json
import requests


KEYWORDS_URL = 'http://api.bosonnlp.com/keywords/analysis'

for i in range(1, 9):

    filename=(r'/Users/JJjie/Downloads/txt5/0%02d.txt'%(i))
    print ('处理%s'%(filename))

    input_file = open(filename)
    text = input_file.read( )
    input_file.close()

    text = text.decode('GBK')

    params = {'top_k': 300}
    data = json.dumps(text)
    headers = {'X-Token': '你的密钥，在控制台最下方'}
    resp = requests.post(KEYWORDS_URL, headers=headers, params=params, data=data.decode('utf-8'))

    filename=(r'/Users/JJjie/Downloads/txt5/word_0%02d.txt'%(i))
    print ('输出%s\n'%(filename))

    output_file=open(filename,'w')
    for weight, word in resp.json():
        output_file.write('%.0f %s\n'%(weight*weight*10000, word))
    output_file.close()