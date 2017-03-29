# -*- coding: utf-8 -*-

import  xdrlib ,sys
import xlrd

def open_excel(file= 'file.xls'):
    try:
        data = xlrd.open_workbook(file)
        return data
    except Exception,e:
        print str(e)

# 根据索引获取Excel表格中的数据   参数:file：Excel文件路径     colnameindex：表头列名所在行的所以  ，by_index：表的索引

def excel_table_byindex(file= 'file.xls',colnameindex=0,by_index=0):
    data = open_excel(file)
    table = data.sheets()[by_index]
    nrows = table.nrows #行数
    ncols = table.ncols #列数
    colnames =  table.row_values(colnameindex) #某一行数据
    list =[]
    for rownum in range(0,nrows):
         row = table.row_values(rownum)
         if row:
             app = []
             for i in range(len(colnames)):
                 app.append(row[i])
             list.append(app)
    return list

#根据名称获取Excel表格中的数据   参数:file：Excel文件路径     colnameindex：表头列名所在行的所以  ，by_name：Sheet1名称
def excel_table_byname(file= 'file.xls',colnameindex=0,by_name=u'Sheet1'):
    data = open_excel(file)
    table = data.sheet_by_name(by_name)
    nrows = table.nrows #行数
    colnames =  table.row_values(colnameindex) #某一行数据
    list =[]
    for rownum in range(0,nrows):
         row = table.row_values(rownum)
         if row:
             app = []
             for i in range(len(colnames)):
                app.append(row[i])
             list.append(app)
    return list

# tuple 规定了读取的文件块：起始坐标（x,y）终点坐标（x,y）
def excel_table_byrow_and_col(file = 'file.xls', by_name=u'Sheet1', *tuple):
    data = open_excel(file)
    table = data.sheet_by_name(by_name)
    list = []
    for rownum in range(tuple[0][0], tuple[1][0]):
        row = table.row_values(rownum)
        if row:
            app = []
            for i in range(tuple[0][1], tuple[1][1]):
                app . append(row[i])
            list.append(app)
    return list

