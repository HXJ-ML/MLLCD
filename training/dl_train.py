import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from Model import MLDoLC
import os
from tqdm import trange

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(file_train, file_val, trans):
    # 读取数据
    dataset_train = datasets.ImageFolder(file_train, transform=trans)
    dataset_val = datasets.ImageFolder(file_val, transform=trans)
    # 导入数据
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)
    print(dataset_train.class_to_idx)
    return train_loader, val_loader


def train(model, loader, optimizer, loss_function, device):
    model.train()
    sum_loss = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        sum_loss += loss.data.item()
    ave_loss = sum_loss / len(loader)
    return ave_loss


def te(model, loader, loss_faction, device):
    model.eval()
    sum_loss = 0
    precision = 0
    recall = 0
    sensitivity = 0
    f1_score = 0
    correct = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_faction(output, target)
            _, pred = torch.max(output.data, 1)
            # print(pred, target)
            correct += (pred == target).sum()
            for b in range(BATCH_SIZE):
                if target[b] == 0:
                    if pred[b] == target[b]:
                        tp = tp + 1
                    else:
                        fn = fn + 1
                else:
                    if pred[b] == target[b]:
                        tn = tn + 1
                    else:
                        fp = fp + 1
            sum_loss += loss.data.item()
        print(' tp{}, tn{}, fp{}, fn{}'.format(tp, tn, fp, fn))
        avg_loss = sum_loss / len(loader)
        acc = correct / len(loader.dataset)
        if tp + fp != 0:
            precision = round(tp / (tp + fp), 4) * 100
        if tp + fn != 0:
            recall = round(tp / (tp + fn), 4) * 100
        if tn + fp != 0:
            sensitivity = round(tn / (tn + fp), 4) * 100
        if tp + fp != 0 or tp + fn != 0:
            f1_score = round((2 * precision * recall) / (precision + recall), 4) * 100

        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {}%, Recall: {}%, Sensitivity: {}%, F1_score: {}%\n'.format(avg_loss, acc * 100, precision, recall,
                                                                                                                                           sensitivity, f1_score))


# The learning rate of meta network
task_lr = 1e-5
BATCH_SIZE = 2
PATCH_SIZE = 1024
PATCH_NUMBER = 60
TASK_EPOCHs = 100
DIM = 4096

transform = transforms.Compose([
    # 旋转
    transforms.RandomRotation(15),
    transforms.RandomRotation(30),
    transforms.RandomRotation(60),
    transforms.RandomRotation(90),
    transforms.RandomRotation(100),
    # 依概率p垂直翻转
    transforms.RandomVerticalFlip(),
    # 依概率p水平翻转
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
net = MLDoLC(
    patch_size=PATCH_SIZE,
    embedding_dim=DIM,
    depth=4,
    heads=8,
    hidden_dim=6400,
    num_classes=2,
    pool='cls',
    batch_size=BATCH_SIZE,
    token_number=PATCH_NUMBER,
    device=DEVICE,
    dim_head=2048,
    dropout=0.5,
    emb_dropout=0.4
)
net = net.to(DEVICE)

loss_func = torch.nn.CrossEntropyLoss().to(DEVICE)

root = '/data/hxjdata/svsdata/'
train_filename = os.path.join(root, 'loss', 'SupportSet_5')
val_filename = os.path.join(root, 'loss', 'QuerySet')
train_loader, val_loader = load_data(train_filename, val_filename, transform)
opt = torch.optim.SGD(net.parameters(), lr=task_lr)
print("--------------------task {} begin training--------------------".format('LSCC'))
for t in trange(TASK_EPOCHs):
    train_loss = train(model=net, loader=train_loader, optimizer=opt, loss_function=loss_func, device=DEVICE)
    print("EPOCH {} test loss{}--------------------".format(t, train_loss))
print("--------------------task {} begin testing--------------------".format('LSCC'))
te(model=net, loader=val_loader, loss_faction=loss_func, device=DEVICE)
