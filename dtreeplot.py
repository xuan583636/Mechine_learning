# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

'''
树的节点类
data:树的数组结构的一项，4值
height:节点的高
'''
class treenode:
    def __init__(self, data, height):
        self.father=data[0]             # 父节点
        self.children=[]                # 子节点列表
        self.data=data[1]               # 节点标签
        self.height=height
        self.pos=0                      # 节点计算时最终位置，计算时只保存相对位置
        self.offset=0                   # 节点最终位置与初始位置的相对值
        self.data_to_father=data[2]     # 链接父节点的属性值
        # 如果有阈值，则加入阈值
        if type(data[3]) != list:
            self.data_to_father = self.data_to_father + str(data[3])

'''
树的绘制类
link:树的数组结构
minspace:节点间的距离
r:节点的绘制半径
lh:层高
'''
class dtreeplot:
    def __init__(self, link, minspace, r, lh):
        s = len(link)
        # 所有节点的列表，第一项为根节点
        treenodelist = []
        # 节点的层次结构
        treelevel=[]

        # 处理树的数组结构
        for i in range(0,s):
            # 根节点的index与其父节点的index相同
            if link[i][0]==i:
                treenodelist.append(treenode(link[i],0))
            else:
                treenodelist.append(treenode(link[i],treenodelist[link[i][0]].height+1))
                treenodelist[link[i][0]].children.append(treenodelist[i])
                treenodelist[i].father=treenodelist[link[i][0]]
            # 如果有新一层的节点则新建一层
            if len(treelevel)==treenodelist[i].height:
                treelevel.append([])
            treelevel[treenodelist[i].height].append(treenodelist[i])

        # 控制绘制图像的坐标轴
        self.right = 0
        self.left = 0
        # 反转层次，从底往上画
        treelevel.reverse()
        # 计算每个节点的位置
        self.__calpos(treelevel, minspace)
        # 绘制树形
        self.__drawtree(treenodelist[0], r, lh, 0)
        plt.xlim(xmin=self.left, xmax=self.right+minspace)
        plt.ylim(ymin=len(treelevel)*lh+lh/2, ymax=lh/2)
        plt.show()

    '''
    逐一绘制计算每个节点的位置
    nodes:节点集合
    l,r:左右区间
    start:当前层的初始绘制位置
    minspace:使用的最小间距
    '''
    def __calonebyone(self,nodes,l,r,start,minspace):
        for i in range(l,r):
                nodes[i].pos=max(nodes[i].pos,start)
                start=nodes[i].pos+minspace;
        return start;

    '''
    计算每个节点的位置与相对偏移
    treelevel：树的层次结构
    minspace:使用的最小间距
    '''
    def __calpos(self,treelevel,minspace):
        #按层次画
        for nodes in treelevel:
            #记录非叶节点
            noleaf=[]
            num=0;
            for node in nodes:
                if len(node.children)>0:
                    noleaf.append(num)
                    node.pos=(node.children[0].pos+node.children[-1].pos)/2
                num=num+1

            start=minspace

            #如果全是非叶节点，直接绘制
            if(len(noleaf))==0:
                self.__calonebyone(nodes,0,len(nodes),0,minspace)
            else:
                start=nodes[noleaf[0]].pos-noleaf[0]*minspace
                self.left=min(self.left,start-minspace)
                start=self.__calonebyone(nodes,0,noleaf[0],start,minspace)
                for i in range(0,len(noleaf)):
                    nodes[noleaf[i]].offset=max(nodes[noleaf[i]].pos,start)-nodes[noleaf[i]].pos
                    nodes[noleaf[i]].pos=max(nodes[noleaf[i]].pos,start)

                    if(i<len(noleaf)-1):
                            #计算两个非叶节点中间的间隔，如果足够大就均匀绘制
                            dis=(nodes[noleaf[i+1]].pos-nodes[noleaf[i]].pos)/(noleaf[i+1]-noleaf[i])
                            start=nodes[noleaf[i]].pos+max(minspace,dis)
                            start=self.__calonebyone(nodes,noleaf[i]+1,noleaf[i+1],start,max(minspace,dis))
                    else:
                         start=nodes[noleaf[i]].pos+minspace
                         start=self.__calonebyone(nodes,noleaf[i]+1,len(nodes),start,minspace)

    '''
    采用先根遍历绘制树
    treenode:当前遍历的节点
    r:半径
    lh:层高
    curoffset:每层节点的累计偏移
    '''
    def __drawtree(self,treenode,r,lh,curoffset):
         #加上当前的累计偏差得到最终位置
         treenode.pos=treenode.pos+curoffset

         if(treenode.pos>self.right):
             self.right=treenode.pos

         #如果是叶节点则画圈，非叶节点画方框
         if(len(treenode.children)>0):
             drawrect(treenode.pos,(treenode.height+1)*lh,r)
             plt.text(treenode.pos, (treenode.height+1)*lh, treenode.data+'=?', color=(0,0,1),ha='center', va='center')
         else:
             drawcircle(treenode.pos,(treenode.height+1)*lh,r)
             plt.text(treenode.pos, (treenode.height+1)*lh, treenode.data, color=(1,0,0),ha='center', va='center')

         num=0;
         #先根遍历
         for node in treenode.children:
             self.__drawtree(node,r,lh,curoffset+treenode.offset)

             #绘制父节点到子节点的连线
             num=num+1

             px=(treenode.pos-r)+2*r*num/(len(treenode.children)+1)
             py=(treenode.height+1)*lh-r-0.02

             #按到圆到方框分开画
             if(len(node.children)>0):
                 px1=node.pos
                 py1=(node.height+1)*lh+r
                 off=np.array([px-px1,py-py1])
                 off=off*r/np.linalg.norm(off)

             else:
                 off=np.array([px-node.pos,-lh+1])
                 off=off*r/np.linalg.norm(off)
                 px1=node.pos+off[0]
                 py1=(node.height+1)*lh+off[1]

             #计算父节点与子节点连线的方向与角度
             plt.plot([px,px1],[py,py1],color=(0,0,0))
             pmx=(px1+px)/2-(1-2*(px<px1))*0.4
             pmy=(py1+py)/2+0.4
             arc=np.arctan(off[1]/(off[0]+0.0000001))
             #绘制文本以及旋转
             plt.text(pmx,pmy, node.data_to_father, color=(1,0,1),ha='center', va='center', rotation=arc/np.pi*180)

'''
画圆
'''
def drawcircle(x,y,r):
     theta = np.arange(0, 2 * np.pi, 2 * np.pi / 1000)
     theta = np.append(theta, [2 * np.pi])
     x1=[]
     y1=[]
     for tha in theta:
         x1.append(x + r * np.cos(tha))
         y1.append(y + r * np.sin(tha))
     plt.plot(x1, y1,color=(0,0,0))

'''
画矩形
'''
def drawrect(x,y,r):
     x1=[x-r,x+r,x+r,x-r,x-r]
     y1=[y-r,y-r,y+r,y+r,y-r]
     plt.plot(x1, y1,color=(0,0,0))