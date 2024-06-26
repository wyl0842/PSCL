import argparse
import os
import ast
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch implementation of ICLR 2022 Oral paper PiCO')

parser.add_argument('--exp-dir', default='experiment/PiCO', type=str,
                    help='experiment directory for saving checkpoints and logs')

args = parser.parse_args()

print(args)

ori_num = np.zeros(10)
noisy_num = np.zeros(10)
correct_num = np.zeros(10)
correction_rate = np.zeros(10)

with open(os.path.join(args.exp_dir, 'correction.log'), 'r') as f:
# with open('correction.log', 'r') as f1:
    while True:
        line = f.readline()     # 逐行读取
        if not line:
            break
        each_line  = line.split(';')
        # 原本标签
        clean_label = int(each_line[0])
        # 噪声标签
        noisy_label_oh = ast.literal_eval(each_line[1])
        noisy_label_oh = np.array(noisy_label_oh)
        noisy_label = np.argmax(noisy_label_oh)
        # 置信度
        confidence_oh = ast.literal_eval(each_line[2])
        confidence_oh = np.array(confidence_oh)
        confidence = np.argmax(confidence_oh)

        ori_num[clean_label] = ori_num[clean_label] + 1
        if noisy_label != clean_label:
            noisy_num[clean_label] = noisy_num[clean_label] + 1
            if confidence == clean_label:
                correct_num[clean_label] = correct_num[clean_label] + 1
        # print(noisy_label)
        # print(confidence)
    
print(ori_num, ori_num.sum())
print(noisy_num, noisy_num.sum())
print(correct_num, correct_num.sum())
print(correct_num/noisy_num, correct_num.sum()/noisy_num.sum())

# with open('correction_select.txt','w') as f2:
#     for line in correction_select:
#         f2.write(line)

# print(correction_select)
# print(len(correction_select))