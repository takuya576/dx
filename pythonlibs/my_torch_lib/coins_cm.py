import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(
    cm,
    classes,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
    save_path=None,
):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        plt.rcParams.update({"font.size": 8})

    plt.clf()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=20)

    # メモリの文字サイズを調整
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            font_size = 7
        else:
            font_size = 7

        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=font_size,  # 数値のフォントサイズを設定
        )

    plt.tight_layout()
    plt.ylabel("True label", fontsize=14)
    plt.xlabel("Predicted label", fontsize=14)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def make_cm(device, epoch, test_loader, save_dir, net):
    classes = [
        str(i1) + str(i2) + str(i3) + str(i4)
        for i1 in range(0, 2)
        for i2 in range(0, 2)
        for i3 in range(0, 2)
        for i4 in range(0, 2)
    ]

    y_preds = []
    y_tests = []

    with torch.no_grad():
        for data in test_loader:
            x_test, y_test = data

            x_test = x_test.to(device)
            y_test = y_test.to(device)

            y_output = net(x_test)

            if len(y_output[0]) == 16:  # 16class
                _, y_pred = torch.max(y_output.data, 1)

            else:  # vector
                outputs_sig = torch.sigmoid(y_output)
                predicted_vec = torch.where(outputs_sig < 0.5, 0, 1)
                y_pred = torch.tensor(
                    [
                        predicted_vec[i][0] * 8
                        + predicted_vec[i][1] * 4
                        + predicted_vec[i][2] * 2
                        + predicted_vec[i][3] * 1
                        for i in range(len(predicted_vec))
                    ]
                ).to(device)
            y_preds.extend(y_pred.tolist())
            y_tests.extend(y_test.tolist())

    confusion_mtx = confusion_matrix(y_tests, y_preds)

    plt.rcParams["axes.grid"] = False

    os.makedirs(os.path.join(f"{save_dir}", "confusion_matrix"), exist_ok=True)

    save_confusion_matrix(
        confusion_mtx,
        classes,
        normalize=False,
        title=f"Confusion Matrix at {epoch}",
        cmap=plt.cm.Reds,
        save_path=os.path.join(
            f"{save_dir}",
            "confusion_matrix",
            f"cm_count_{epoch}.png",
        ),
    )

    save_confusion_matrix(
        confusion_mtx,
        classes,
        normalize=True,
        title=f"Confusion Matrix at {epoch}",
        cmap=plt.cm.Reds,
        save_path=os.path.join(
            f"{save_dir}",
            "confusion_matrix",
            f"cm_count_{epoch}_norm.png",
        ),
    )

    plt.rcParams["axes.grid"] = True
