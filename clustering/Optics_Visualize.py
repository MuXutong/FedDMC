from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib
import warnings

from clustering.Optics import OPTICS1, plotReachability

warnings.filterwarnings('ignore')

# folder_str = r'H:\By-FL\clustering\r65_grad_data'
folder_str = r'H:\By-FL\logs\2022-11-01\16.24.55\gradient'
folder = pathlib.Path.cwd().parent.joinpath(folder_str)

createVars = locals()  # 以字典类型返回当前位置所有局部变量，后续DataFrame
for fp in folder.iterdir():  # 迭代文件夹
    if fp.match('*.csv'):  # re正则匹配判断文件夹里是否有csv文件
        varname = fp.parts[-1].split('.')[0]  # 按照‘.’的方式切割，取-1，得到csv文件的名字
        # createVars[varname] = pd.read_csv(fp)#添加文件，转为pandas的DataFrame
        print(varname)#打印文件名

        data = pd.read_csv(fp, header=None)
        data = np.array(data)
        data = data[:, :3]



        OP = OPTICS1(data, minPts=3)
        OP.train()
        OP.predict()

        result = OP.reach_dists[OP.orders]
        eps = np.mean(result)*1.5
        fig = plt.figure()
        plt.plot(range(0, len(result)), result)
        plt.plot([0, len(result)], [eps, eps])
        plt.show()
        fig.savefig(folder_str + '/Optics_' + varname + '.png')