import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader

from Model import MLDoLC
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(file_train, file_val, trans):
    # 读取数据
    dataset_train = datasets.ImageFolder(file_train, transform=trans)
    dataset_val = datasets.ImageFolder(file_val, transform=trans)
    # 导入数据
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)
    print(dataset_train.class_to_idx)
    return train_loader, val_loader


def train(model, loader, optimizer, loss_function, device):
    model.train()
    sum_loss = 0
    print(len(loader))
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(output)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        sum_loss += loss.data.item()
    ave_loss = sum_loss / len(loader)
    return ave_loss


def val(model, loader, optimizer, loss_faction, device, grad):
    model.eval()
    sum_loss = 0
    correct = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # print(output)
        optimizer.zero_grad()
        loss = loss_faction(output, target)
        # loss.requires_grad_(True)
        loss.backward()
        for n, p in net.named_parameters():
            grad[n] = grad[n] + p.grad
        # optimizer.step()
        pred = output.argmax(1)
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
    ave_loss = sum_loss / len(loader)
    return ave_loss


def te(model, loader, loss_faction, device, epoch, ith):
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
    score_list = []
    label_list = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            score_list.extend(output.detach().cpu().numpy())
            label_list.extend(target.cpu().numpy())

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
        if not (precision == 0 and recall == 0):
            f1_score = round((2 * precision * recall) / (precision + recall), 4)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {}%, Recall: {}%, Sensitivity: {}%, F1_score: {}%\n'.format(avg_loss, acc * 100, precision, recall,
                                                                                                                                           sensitivity, f1_score))
        score_array = np.array(score_list)
        # 将label转换成onehot形式
        label_tensor = torch.tensor(label_list)
        label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
        label_onehot = torch.zeros(label_tensor.shape[0], 2)
        label_onehot.scatter_(dim=1, index=label_tensor, value=1)
        label_onehot = np.array(label_onehot)

        # 调用sklearn库，计算每个类别对应的fpr和tpr
        fpr_dict = dict()
        tpr_dict = dict()
        roc_auc_dict = dict()
        for i in range(0, 2):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
            roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        # abnormal
        fpr_dict["abnormal"], tpr_dict["abnormal"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
        roc_auc_dict["abnormal"] = auc(fpr_dict["abnormal"], tpr_dict["abnormal"])

        # normal
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(2)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(0, 2):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        # Finally average it and compute AUC
        mean_tpr /= 2
        fpr_dict["normal"] = all_fpr
        tpr_dict["normal"] = mean_tpr
        roc_auc_dict["normal"] = auc(fpr_dict["normal"], tpr_dict["normal"])

        # 绘制所有类别平均的roc曲线
        plt.figure()
        lw = 2
        plt.plot(fpr_dict["abnormal"], tpr_dict["abnormal"],
                 label='abnormal-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc_dict["abnormal"]),
                 color='blue', linestyle=':', linewidth=4)

        plt.plot(fpr_dict["normal"], tpr_dict["normal"],
                 label='normal-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc_dict["normal"]),
                 color='yellow', linestyle=':', linewidth=4)
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC of  20-shots about classifying')
        plt.legend(loc="lower right")
        plt.savefig('/root/autodl-tmp/MLDoLC/training/roc/{}_roc_20shot_{}.png'.format(ith, epoch))
        plt.close()
        return avg_loss, acc


# The learning rate of meta network
meta_lr = 1e-4
task_lr = 1e-2
BATCH_SIZE = 5
PATCH_SIZE = 256
PATCH_NUMBER = 60 * 16
MATA_EPOCHs = 500
TASK_EPOCHs = 10
TEST_EPOCHs = 30
EMBEDDING_DIM = 1000
HIDDEN_DIM = 600
ITH = 4
print("meta_lr {}, task_lr {}, 20 shots,{} th,patch size{}".format(meta_lr, task_lr, ITH,PATCH_SIZE))

transform = transforms.Compose([
    # transforms.RandomRotation(15),
    # transforms.RandomRotation(45),
    # transforms.RandomRotation(75),
    # transforms.RandomRotation(175),
    transforms.ToTensor(),
])

net = MLDoLC(
    patch_size=PATCH_SIZE,
    embedding_dim=EMBEDDING_DIM,
    depth=6,
    heads=4,
    hidden_dim=HIDDEN_DIM,
    num_classes=2,
    pool='cls',
    batch_size=BATCH_SIZE,
    token_number=PATCH_NUMBER,
    device=DEVICE,
    dim_head=EMBEDDING_DIM,
    dropout=0.,
    emb_dropout=0.
)
net = net.to(DEVICE)

loss_func = torch.nn.CrossEntropyLoss().to(DEVICE)

root = '/root/autodl-tmp/svsdata/'
dataSet_list = ['CCRCC', 'CM', 'PDA', 'SAR', 'UCEC', 'LSCC']
for m in range(MATA_EPOCHs):
    print("-----------------EPOCH {}----------------------------".format(m))
    torch.save(obj=net.state_dict(), f="/root/autodl-tmp/MLDoLC/pth/{}_meta_parameters_20_{}.pth".format(ITH, m))
    total_grad = {}
    for name, param in net.named_parameters():
        total_grad[name] = torch.zeros_like(param)
    # Meta train
    print("---------------------------------------------META TRAINING-----------------------------------------------------------")
    for i in range(0, len(dataSet_list) - 1):
        train_filename = os.path.join(root, dataSet_list[i], 'SupportSet_20')
        val_filename = os.path.join(root, dataSet_list[i], 'QuerySet')
        task_train_loader, task_val_loader = load_data(train_filename, val_filename, transform)

        net.load_state_dict(torch.load("/root/autodl-tmp/MLDoLC/pth/{}_meta_parameters_20_{}.pth".format(ITH, m)))
        task_opt = torch.optim.SGD(net.parameters(), lr=task_lr)
        print("--------------------task {} begin training--------------------".format(dataSet_list[i]))
        for t in range(TASK_EPOCHs):
            train_loss = train(model=net, loader=task_train_loader, optimizer=task_opt, loss_function=loss_func, device=DEVICE)
            print("TASK_EPOCHs {} task {}    train loss{}--------------------".format(t, dataSet_list[i], train_loss))
        print("--------------------task {} begin testing--------------------".format(dataSet_list[i]))
        val_loss = val(model=net, loader=task_val_loader, optimizer=task_opt, loss_faction=loss_func, device=DEVICE, grad=total_grad)
        print("task {}     test loss{}--------------------".format(dataSet_list[i], val_loss))

    # Meta update
    net.load_state_dict(torch.load("/root/autodl-tmp/MLDoLC/pth/{}_meta_parameters_20_{}.pth".format(ITH, m)))
    for name, param in net.named_parameters():
        param = param - meta_lr * (total_grad[name])
    torch.save(obj=net.state_dict(), f="/root/autodl-tmp/MLDoLC/pth/{}_meta_parameters_20_{}.pth".format(ITH, m + 1))

    # Test
    print("---------------------------------------------TESTING-----------------------------------------------------------")
    net.load_state_dict(torch.load("/root/autodl-tmp/MLDoLC/pth/{}_meta_parameters_20_{}.pth".format(ITH, m + 1)))
    test_filename = os.path.join(root, dataSet_list[5], 'SupportSet_20')
    val_filename = os.path.join(root, dataSet_list[5], 'QuerySet')
    test_train_loader, test_val_loader = load_data(test_filename, val_filename, transform)
    task_opt = torch.optim.SGD(net.parameters(), lr=task_lr)
    print("--------------------task {} begin training--------------------".format(dataSet_list[5]))
    for t in range(TEST_EPOCHs):
        train_loss = train(model=net, loader=test_train_loader, optimizer=task_opt, loss_function=loss_func, device=DEVICE)
        print("------------task {} begin testing--------------------".format(dataSet_list[5]))
        Loss, Acc = te(model=net, loader=test_val_loader, loss_faction=loss_func, device=DEVICE, epoch=m, ith=ITH)
