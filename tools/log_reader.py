import numpy as np
from matplotlib import pyplot as plt

# loss_names = ["G", "G_A", "G_B", "seg", "D_A", "D_B", "G_A_idt", "G_A_rec", "G_A_gan", "G_B_idt", "G_B_rec", "G_B_gan"]
loss_names = ["G_A", "G_B", "seg", "D_A", "D_B"]


def preprocess(data):
    for loss_name in data.keys():
        if loss_name not in loss_names:
            continue

        arr = np.array(data[loss_name])
        arr = (arr - min(arr)) / (max(arr) - min(arr))
        data[loss_name] = arr


def draw_graph(data):
    y_axis = None
    for loss_name in data.keys():
        if loss_name not in loss_names:
            continue
        if y_axis is None:
            y_axis = list(range(len(data[loss_name])))

        plt.plot(y_axis, data[loss_name], label=loss_name)

    plt.ylim([-0.2, 1.2])
    plt.legend()
    plt.show()


def read_log(log_path):
    train_loss = {}
    val_loss = {}
    with open(log_path, 'r') as rf:
        for line in rf.readlines()[1:]:
            line = line.replace('\n', '')
            line_split = line.split(']')
            if len(line_split) == 1:
                # val line
                line_split = line_split[0][13:].split('  ')
                target_dict = val_loss
            elif len(line_split) == 2:
                # train total line
                line_split = line_split[1][2:].split('  ')
                target_dict = train_loss
            elif len(line_split) == 3:
                # train iter line
                continue

            for attr in line_split:
                loss_name, loss = attr.split(': ')
                if loss_name == 'time':
                    continue
                if loss_name not in loss_names:
                    continue
                if loss_name not in target_dict:
                    target_dict[loss_name] = []
                target_dict[loss_name].append(float(loss))

    return train_loss, val_loss


if __name__ == '__main__':
    log_path = '../results/20220609170613_stackedSeg_default_weight/log.txt'

    train_loss, val_loss = read_log(log_path)

    preprocess(val_loss)
    draw_graph(val_loss)
