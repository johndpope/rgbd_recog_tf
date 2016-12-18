import matplotlib.pyplot as plt
import sys, ipdb


#f = '../logs/training_rgb.log'


def main(pth):
    content = open(pth, 'r').read().splitlines()
    loss = []
    train_precision = []
    eval_precision = []
    for i in range(len(content)):
        line = content[i]
        if "loss" in line:
            l = float(line[line.find('=')+2 : line.find('(')-1])
            loss.append(l)

        if "Training data eval" in line:
            if i+1 >= len(content): break
            line = content[i+1]
            p = float(line[line.find('Precision:')+10:])
            train_precision.append(p)

        if "Validation data eval" in line:
            if i+1 >= len(content): break
            line = content[i+1]
            p = float(line[line.find('Precision:')+10:])
            eval_precision.append(p)
        i += 1

    plt.figure()
    plt.plot(train_precision, 'g', label='train precision')
    plt.plot(eval_precision, 'b', label='eval precision')
    plt.legend(loc=4)
    plt.grid('on')
    plt.title('Precision')

    plt.figure()
    plt.semilogy(loss)
    plt.grid('on')
    plt.title('Loss')


if __name__ == '__main__':
    for pth in sys.argv[1:]:
        main(pth)
    plt.show()
