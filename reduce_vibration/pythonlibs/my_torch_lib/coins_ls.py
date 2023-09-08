import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap


def make_ls(device, epoch, test_loader, save_dir, net):
    y_tests = []
    y_outputs = []

    with torch.no_grad():
        for data in test_loader:
            x_test, y_test = data

            x_test = x_test.to(device)
            y_test = y_test.to(device)

            y_output = net(x_test)

            y_outputs.extend(y_output.tolist())
            y_tests.extend(y_test.tolist())

    reducer = umap.UMAP(random_state=42)
    reducer.fit(y_outputs)
    embedding = reducer.transform(y_outputs)

    plt.clf()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=y_tests, cmap="Spectral", s=5)
    plt.gca().set_aspect("equal", "datalim")
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    plt.title(
        f"UMAP projection of the output(fc) features @ epoch={epoch:d}", fontsize=12
    )
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    os.makedirs(os.path.join(f"{save_dir}", "latent_space"), exist_ok=True)
    plt.savefig(
        os.path.join(
            f"{save_dir}",
            "latent_space",
            f"ls_fc_{epoch}.png",
        ),
        bbox_inches="tight",
    )
