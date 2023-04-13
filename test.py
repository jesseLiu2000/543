import torch
import torch.nn as nn
import torch.utils.data as Data


# 模型结构
class ClassifyModel(nn.Module):
    def __init__(self, input_dim, hiden_dim, output_dim):
        super(ClassifyModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hiden_dim)
        self.linear2 = nn.Linear(hiden_dim, output_dim)

    def forward(self, x):
        hidden = self.linear1(x)
        activate = torch.relu(hidden)
        output = self.linear2(activate)
        # 注意：整个模型结构的最后一层是线性全连接层，并非是sigmoid层，是因为之后直接接CrossEntropy()损失函数，已经内置了log softmax层的过程了
        # 若损失函数使用NLLLoss()则需要在模型结构中先做好tanh或者log_softmax
        # 即：y^ = softmax(x), loss = ylog(y^) + (1-y)log(1-y^)中的过程

        return output


def get_acc(outputs, labels):
    """计算acc"""
    _, predict = torch.max(outputs.data, 1)
    total_num = labels.shape[0]*1.0
    correct_num = (labels == predict).sum().item()
    acc = correct_num / total_num

    return acc


# 准备数据
x = torch.unsqueeze(torch.linspace(-10, 10, 50), 1)  # 50*1
y = torch.cat((torch.ones(25), torch.zeros(25))).type(torch.LongTensor)   # 1*50

dataset = Data.TensorDataset(x, y)
dataloader = Data.DataLoader(dataset=dataset, batch_size=5, shuffle=True)

# 模型实例化
# 方式一：
model = ClassifyModel(1, 10, 2)

# 方式二：直接构建模型序列
model2 = torch.nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10,2)
)



# 优化器
optim = torch.optim.Adam(model2.parameters(), lr=0.0001)

# 损失
loss_fun = nn.CrossEntropyLoss()

# 训练
for e in range(1000):
    epoch_loss = 0
    epoch_acc = 0
    for i, (x, y) in enumerate(dataloader):
        optim.zero_grad()

        out = model2(x)
        # print("out", out.size())
        # print("y", y.size())
        loss = loss_fun(out, y)

        loss.backward()
        optim.step()

        epoch_loss += loss.data
        epoch_acc += get_acc(out, y)

    if e % 200 == 0:
        print('epoch: %d, loss: %f, acc: %f' % (e, epoch_loss / 50, epoch_acc / 50))


# # 保存与加载模型
# 方式1
# model_path = 'data/classify_model.pkl'
# torch.save(model, model_path)
# reload_model = torch.load(model_path)
# print(reload_model(torch.Tensor([5])).data)
#
# # # 保存与加载模型
# # 方式2
# model_path2 = 'data/classify_model2.pkl'
# torch.save(model.state_dict(), model_path2)
# reload_model = ClassifyModel(1,10,2)
# reload_model.load_state_dict(torch.load(model_path2))
# print(reload_model(torch.Tensor([5])).data)
