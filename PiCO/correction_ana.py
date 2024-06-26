import argparse
import os
import ast
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch implementation of ICLR 2022 Oral paper PiCO')

parser.add_argument('--exp-dir', default='experiment/PiCO', type=str,
                    help='experiment directory for saving checkpoints and logs')

args = parser.parse_args()

print(args)

correction_select = []

# with open(os.path.join(args.exp_dir, 'correction.log'), 'r') as f:
with open('correction.log', 'r') as f1:
    while True:
        line = f1.readline()     # 逐行读取
        if not line:
            break
        each_line  = line.split(';')
        noisy_label_oh = ast.literal_eval(each_line[0])
        noisy_label_oh = np.array(noisy_label_oh)
        noisy_label = np.argmax(noisy_label_oh)
        confidence_oh = ast.literal_eval(each_line[1])
        confidence_oh = np.array(confidence_oh)
        confidence = np.argmax(confidence_oh)
        if noisy_label != confidence:
            correction_select.append(line)
        # print(noisy_label)
        # print(confidence)

with open('correction_select.txt','w') as f2:
    for line in correction_select:
        f2.write(line)

# print(correction_select)
# print(len(correction_select))