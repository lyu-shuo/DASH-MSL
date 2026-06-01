# DASH-MSL
DASH-MSL: Unveiling Vulnerabilities of Multi-Client Split learning to Poisoning Attacks on Shared Weights

![Image](https://github.com/user-attachments/assets/3b42b821-21f2-48ca-b295-79ce54b2aa52)
## Code
This repository contains all the code needed to run the Dash-MSL. 

The implementation is based on **Pytorch**.

## 文件说明
AttFunc.py, Fisher_LeNet.py, Fisher_SA.py 为方案中的攻击函数（扰动函数，修改权重值（即投毒），攻击系数优化）
clients_datasets.py 加载数据
models.py 模型
utils.py 工具类

以上文件无需改动，在实验代码中调用即可

exper1.ipynb 动机实验
Exper-ID 攻击者处于队列中不同位置的实验
Exper-Alpha 不同扰动（投毒）比例的实验
Exper-Strategy 不同攻击策略的实验

