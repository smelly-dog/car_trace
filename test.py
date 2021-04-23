import torch
import torch.nn as nn
import math

loss = nn.CrossEntropyLoss()
input = torch.randn(1, 5, requires_grad= True)
target = torch.empty(1, dtype = torch.long).random_(5)
output = loss(input, target)

print("input {}".format(input.shape))
print("target {}".format(target.shape))

print('输入为5类：', input)
print('要计算loss的真实类别', target)
print('loss=', output)
#自己计算的结果
first = 0
for i in range(1):
    first -= input[i][target[i]]
second = 0
for i in range(1):
    for j in range(5):
        second += math.exp(input[i][j])
res = 0
res += first + math.log(second)
print('自己的计算结果：', res)