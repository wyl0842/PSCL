import re
import matplotlib.pyplot as plt

def extract_accuracy(log_file):
    accuracies = []
    line_count = 0

    with open(log_file, 'r') as file:
        for line in file:
            line_count += 1
            if line_count % 2 == 0:  # 仅处理偶数行
                accuracy_line = re.search(r'Acc\s+(\d+\.\d+)', line)
                if accuracy_line:
                    accuracy = float(accuracy_line.group(1))
                    accuracies.append(accuracy)

    return accuracies

def plot_accuracy_curves(log_files):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 设置不同曲线的颜色
    # line_styles = ['-', '--', '-.', ':']  # 设置不同曲线的线型
    line_styles = ['-', '-']  # 设置不同曲线的线型
    title = ['CL', 'SCL']  # 设置不同曲线的线型

    plt.figure(figsize=(6, 4))
    plt.grid(True, alpha=0.5, linestyle='--', linewidth=0.5)  # 添加网格线

    for i, log_file in enumerate(log_files):
        accuracies = extract_accuracy(log_file)
        generations = range(1, len(accuracies) + 1)

        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        # label = f'Log File {i+1}'
        label = title[i % len(title)]

        plt.plot(generations, accuracies, color=color, linestyle=line_style, linewidth=1, label=label)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # plt.title('Accuracy Change')
    plt.legend(loc='lower right', fontsize='small')

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.tight_layout()

    plt.savefig('accuracy_curves.png', dpi=300)  # 保存图片，设置dpi以提高分辨率
    plt.show()

log_files = ['/home/wangyl/Code/PSCL/PiCO_CL/experiment/PSCL231015dist-CIFAR-10-MOCO/ds_cifar10_nr_0.4_nt_symmetric_lr_0.01_ep_800_ps_80_lw_0.1_pm_0.99_arch_resnet18_heir_False_sd_1/result.log', '/home/wangyl/Code/PSCL/PiCO_CL/experiment/PSCL230531-CIFAR-10/ds_cifar10_nr_0.4_nt_symmetric_lr_0.01_ep_800_ps_80_lw_0.1_pm_0.99_arch_resnet18_heir_False_sd_1/result.log']  # 替换为你的log文件路径

plot_accuracy_curves(log_files)