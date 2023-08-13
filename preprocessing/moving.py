import os
import shutil
from shutil import move, copy, copyfile

# Data root path
root = '/data/hxjdata/svsdata'
# DataSet type
# dataSet_list = ['CCRCC', 'CM', 'PDA', 'SAR', 'UCEC', 'LSCC']
dataSet_list = ['LUAD']
# Support set or query set
s_or_q_list = ['SupportSet_5', 'SupportSet_10', 'SupportSet_20', 'QuerySet']
# abnormal or normal
class_list = ['abnormal', 'normal']
# for i in range(0, len(dataSet_list)):
#     for j in range(len(class_list)):
#         src_path = os.path.join(root, dataSet_list[i], 'SupportSet_50',class_list[j])
#         listdir = os.listdir(src_path)
#         # 5 shots
#         des_path = os.path.join(root, dataSet_list[i], 'SupportSet_20', class_list[j])
#         for k in range(0,20):
#             shutil.copy2(os.path.join(src_path,listdir[k]), des_path)

for i in range(0, len(dataSet_list)):
    for j in range(1, len(class_list)):
        src_path = os.path.join(root, dataSet_list[i], 'SupportSet_5', class_list[j])
        listdir = os.listdir(src_path)
        # 5 shots
        des_path = os.path.join(root, dataSet_list[i], 'SupportSet_50', class_list[j])
        for k in range(0,50):
            move(os.path.join(src_path,listdir[k]), des_path)


# path1 =  os.path.join(root,dataSet_list[0],'wsi',class_list[0],'C3L-00966-21.svs')
# path2 =  os.path.join(root,dataSet_list[0],'wsi',class_list[0],'C3L-00966-22.svs')
# path3 = os.path.join(root,dataSet_list[0],'wsi_50','a0')
# shutil.copy2(path2, path3)

# for i in range(0, len(dataSet_list)):
#     txt_path = os.path.join(root, dataSet_list[i], dataSet_list[i] + '_normal.txt')
#     src_path = os.path.join(root, dataSet_list[i], 'wsi_50')
#     dst_path_ab = os.path.join(root, dataSet_list[i], 'wsi', class_list[0])
#     dst_path = os.path.join(root, dataSet_list[i], 'wsi', class_list[1])

    # # Distinguishing between normal and abnormal data
    # for line in open(txt_path):
    #     file_path = os.path.join(src_path, line[0:12]+'.svs')
    #     move(file_path, dst_path)
    #

    # path = os.path.join(src_path, 'normal')
    # listdir = os.listdir(path)
    # path0 = os.path.join(src_path, 'n9')
    # for j in range(0,5):
    #     file_path = os.path.join(path, listdir[j])
    #     move(file_path, path0)



    # # Building a support set with 5 shots
    # support5_ab_path = os.path.join(root, dataSet_list[i], s_or_q_list[0], class_list[0])
    # support5_path = os.path.join(root, dataSet_list[i], s_or_q_list[0], class_list[1])
    # # abnormal
    # listdir_dstab = os.listdir(dst_path_ab)
    # for t in range(5):
    #     path1 = os.path.join(dst_path_ab, listdir_dstab[t])
    #     shutil.copy2(path1,support5_ab_path)
    # # normal
    # listdir_dst = os.listdir(dst_path)
    # for p in range(5):
    #     path2 = os.path.join(dst_path, listdir_dst[p])
    #     shutil.copy2(path2, support5_path)


    # # abnormal
    # listdir_dstab = os.listdir(dst_path_ab)
    # for t in range(50):
    #     path1 = os.path.join(dst_path_ab, listdir_dstab[t])
    #     path2 = os.path.join(src_path,'abnormal')
    #     shutil.copy2(path1,path2)
    # # normal
    # listdir_dst = os.listdir(dst_path)
    # for p in range(50):
    #     path1 = os.path.join(dst_path, listdir_dst[p])
    #     path2 = os.path.join(src_path,'normal')
    #     shutil.copy2(path1, path2)


