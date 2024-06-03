from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib
import warnings

# samples = pd.read_csv(r'G:\FL\By-FL\clustering\r45_grad_data\n2_r45.csv', header=None)
# samples = np.array(samples)
#
# mergings = linkage(samples, method='complete')
#
# fig = plt.figure(figsize=(10, 6))
# dendrogram(mergings,
#            leaf_rotation=90,
#            leaf_font_size=6, )
# plt.show()

warnings.filterwarnings('ignore')

# folder_str = r'H:\By-FL\clustering\r45_grad_data'
folder_str = r'H:\By-FL\logs\2022-11-01\16.24.55\gradient'
folder = pathlib.Path.cwd().parent.joinpath(folder_str)

createVars = locals()  # 以字典类型返回当前位置所有局部变量，后续DataFrame
for fp in folder.iterdir():  # 迭代文件夹
    if fp.match('*.csv'):  # re正则匹配判断文件夹里是否有csv文件
        varname = fp.parts[-1].split('.')[0]  # 按照‘.’的方式切割，取-1，得到csv文件的名字
        # createVars[varname] = pd.read_csv(fp)#添加文件，转为pandas的DataFrame
        print(varname)#打印文件名

        samples = pd.read_csv(fp, header=None)
        samples = np.array(samples)[:10]
        data = samples[:, :1]

        mergings = linkage(samples, method='average')

        fig = plt.figure(figsize=(15, 10))
        dendrogram(mergings,
                   leaf_rotation=90,
                   leaf_font_size=8, )
        plt.show()
        fig.savefig(folder_str + '/brich_' + varname + '.png')