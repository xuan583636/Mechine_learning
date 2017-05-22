# -*- coding: utf-8 -*-
# @Time    : 2017/5/22 11:20
# @Author  : UNE
# @Project : Mechine_learning
# @File    : Tkinter_GUI_Dtree.py
# @Software: PyCharm

# 《机器学习实战》第九章

import Tkinter as tk
import numpy as np
import CART as rt

import matplotlib
matplotlib.use('TkAgg')     # 所选GUI框架上调用Agg(有个C++的库，用于从图像创建光栅图)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def reDraw(tols, toln):
	reDraw.f.clf()
	reDraw.a = reDraw.f.add_subplot()
	if chkBtnVar.get():
		if toln < 2:
			toln = 2
		myTree = rt.createTree(rt.rawDat, rt.modelLeaf, rt.modelErr, (tols, toln))
	else:
		myTree = rt.createTree(reDraw.rawDat, ops=(tols, toln))
	reDraw.a.scatter(reDraw.rawDat[:, 0], reDraw.rawDat[:, 1], s=5)
	reDraw.canvas.show()

def getInputs():
	try:
		toln = int(tolNentry.get())
	except:
		toln = 10
		print "enter Interger for toln"
		tolNentry.delete(0, END)
		tolNentry.insert(0, '10')
	try:
		tols = int(tolNentry.get())
	except:
		tols = 1.0
		print "enter Float for tols"
		tolNentry.delete(0, END)
		tolNentry.insert(0, '1.0')
	return toln, tols

def drawNewTree():
	toln, tols = getInputs()
	reDraw(tols, toln)


if __name__ == '__main__':
	root = tk.Tk()
	tk.Label(root, text="Plot Place Holder").grid(row=0, columnspan=3)
	tk.Label(root, text="tolN").grid(row=1, column=0)
	tolNentry = tk.Entry(root)
	tolNentry.grid(row=1, column=1)
	tolNentry.insert(0, '10')
	tk.Label(root, text="tolS").grid(row=2, column=0)
	tolNentry = tk.Entry(root)
	tolNentry.grid(row=2, column=1)
	tolNentry.insert(0, '1.0')
	tk.Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)
	chkBtnVar = tk.IntVar()
	chkBtn = tk.Checkbutton(root, text='Model Tree', variable = chkBtnVar)
	chkBtn.grid(row=3, column=0, columnspan=2)

	filename = "/Users/JJjie/Desktop/Projects/dataset/MLiA/ex9_sine.txt"
	reDraw.f = Figure(figsize=(5, 4), dpi=100)
	reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
	reDraw.canvas.show()
	reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

	reDraw.rawDat = np.mat(rt.loadDataSet(filename))
	reDraw.testDat = np.arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
	reDraw(1.0, 10)
	root.mainloop()