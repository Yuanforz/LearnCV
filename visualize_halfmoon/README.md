# 激活函数与非线性性 
心血来潮想到relu+MLP=分段线性函数，想要可视化考察一下不同激活函数对神经网络的非线性表达能力有和影响。 
## 尝试环境 
- pytorch  
- halfmoon数据集
- 2->10->10->1 固定架构  
- AdamW优化器 
## 尝试组合 
成功拟合： 
- abs x abs 
- exp x exp 
- exp x cos 
- sin x sin 
- x^2  
- relu 
- sigmoid 
- sort 
- permute   

无法拟合： 
- *1/x x 1/x*
- *tan x tan
- *sgn*
## 尝试结果 
具体结果见result文件夹。 
令我比较惊讶的是对上一层输入的排序也能带来非线性性，根据上一层输入的大小关系的permute也可收敛。 
激活函数主要决定网络的泛化方向，而对训练数据的拟合能力主要来自于反向传播算法。  