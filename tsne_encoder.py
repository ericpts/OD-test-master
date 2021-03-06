import numpy as np
import csv, os
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import global_vars as Global
from sklearn.manifold import TSNE
from datasets.NIH_Chest import NIHChestBinaryTrainSplit
import seaborn as sns
import argparse
import os
import models as Models
from easydict import EasyDict
import _pickle
from datasets.NIH_Chest import NIHChest
from datasets.PADChest import (
    PADChestBinaryTrainSplit,
    PADChestBinaryValSplit,
    PADChestBinaryTestSplit,
    PADChestSV,
)
import umap

All_OD1 = [
    "UniformNoise",
    #'NormalNoise',
    "MNIST",
    "FashionMNIST",
    "NotMNIST",
    #'CIFAR100',
    "CIFAR10",
    "STL10",
    "TinyImagenet",
    "MURAHAND",
    #'MURAWRIST',
    #'MURAELBOW',
    "MURAFINGER",
    #'MURAFOREARM',
    #'MURAHUMERUS',
    #'MURASHOULDER',
]
ALL_OD2_NIH = ["PADChestAP", "PADChestL", "PADChestAPHorizontal", "PADChestPED"]
ALL_OD2_PAD = ["PADChestAP", "PADChestPA", "PADChestAPHorizontal", "PADChestPED"]
d3_tags_NIH = ["Cardiomegaly", "Pneumothorax", "Nodule", "Mass"]
d3_tags_PAD = ["cardiomegaly", "pneumothorax", "nodule", "mass"]


def proc_data(args, model, D1, d2s, tags):
    Out_X = []
    Out_Y = []
    Cat2Y = {}
    for y, D2 in enumerate(d2s):
        Cat2Y[tags[y]] = y + 1
        loader = DataLoader(D2, shuffle=True, batch_size=args.points_per_d2)
        for i, (X, _) in enumerate(loader):
            x = X.numpy()
            Out_X.append(x)
            Out_Y.append(np.ones(x.shape[0]) * (y + 1))
            break

    Out_X = np.concatenate(Out_X, axis=0)
    Out_Y = np.concatenate(Out_Y, axis=0)
    N_out = Out_X.shape[0]
    print(N_out)
    N_in = max(int(N_out * 0.2), args.points_per_d2)
    In_X = []
    for i in range(N_in):
        In_X.append(D1[i][0].numpy())
    In_Y = np.zeros(N_in)
    print(N_in, len(In_X), len(In_Y))
    Cat2Y["In-Data"] = 0
    ALL_X = np.concatenate((In_X, Out_X))
    ALL_Y = np.concatenate((In_Y, Out_Y))

    new_dataset = TensorDataset(torch.tensor(ALL_X))
    loader = DataLoader(new_dataset, batch_size=64)
    ALL_EMBS = []
    for i, (X,) in enumerate(loader):
        x = model.encode(X.cuda()).data.cpu().numpy()
        ALL_EMBS.append(x)
    ALL_EMBS = np.concatenate(ALL_EMBS, axis=0)

    return ALL_EMBS, ALL_Y, Cat2Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path", type=str, help="unique path for symlink to dataset"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed. (default 42)"
    )
    parser.add_argument(
        "--exp",
        "--experiment_id",
        type=str,
        default="test",
        help="The Experiment ID. (default test)",
    )
    parser.add_argument("--embedding_function", type=str, default="VAE")
    parser.add_argument("--dataset", type=str, default="nihcc")
    parser.add_argument("--encoder_loss", type=str, default="bce")
    # parser.add_argument('--model_path', type=str, default="model_ref/Generic_VAE.HClass/NIHCC.dataset/BCE.max.512.d.12.nH.1024/model.best.pth")
    parser.add_argument("--umap", action="store_true")
    parser.add_argument("--points_per_d2", type=int, default=1024)
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="learning rate in tsne mode, min_dist in umap mode",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=100.0,
        help="perplexity in tsne mode, n_neighbor in umap mode",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1000,
        help="n iter in tsne mode, n epoch in umap mode",
    )
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--plot_percent", default=0.5, type=float)
    args = parser.parse_args()
    args.experiment_id = args.exp

    exp_data = []
    workspace_path = os.path.abspath("workspace")

    exp_list = args.experiment_id.split(",")
    exp_paths = []
    for exp_id in exp_list:
        experiments_path = os.path.join(workspace_path, "experiments", exp_id)
        if not os.path.exists(experiments_path):
            os.makedirs(experiments_path)

        # Make the experiment subfolders.
        for folder_name in exp_data:
            if not os.path.exists(os.path.join(experiments_path, folder_name)):
                os.makedirs(os.path.join(experiments_path, folder_name))
        exp_paths.append(experiments_path)

    if len(exp_list) == 1:
        args.experiment_path = exp_paths[0]
    else:
        print("Operating in multi experiment mode.", "red")
        args.experiment_path = exp_paths

    #####################################################################################################
    if not args.load or not os.path.exists(
        os.path.join(
            args.experiment_path,
            "all_embs_UC3_ppd_%d_d1_%s.npy" % (args.points_per_d2, args.dataset),
        )
    ):
        assert args.dataset in ["NIHCC", "PADChest"]
        if args.dataset.lower() == "nihcc":
            D164 = NIHChestBinaryTrainSplit(
                root_path=os.path.join(args.root_path, "NIHCC"), downsample=64
            )
        elif args.dataset.lower() == "padchest":
            D164 = PADChestBinaryTrainSplit(
                root_path=os.path.join(args.root_path, "PADChest"),
                binary=True,
                downsample=64,
            )

        D1 = D164.get_D1_train()

        emb = args.embedding_function.lower()
        assert emb in ["vae", "ae", "ali"]
        dummy_args = EasyDict()
        dummy_args.exp = "foo"
        dummy_args.experiment_path = args.experiment_path
        if args.encoder_loss.lower() == "bce":
            tag = "BCE"
        else:
            tag = "MSE"
        if emb == "vae":
            model = Global.dataset_reference_vaes[args.dataset][0]()
            home_path = Models.get_ref_model_path(
                dummy_args,
                model.__class__.__name__,
                D164.name,
                suffix_str=tag + "." + model.netid,
            )
            model_path = os.path.join(home_path, "model.best.pth")
        elif emb == "ae":
            model = Global.dataset_reference_autoencoders[args.dataset][0]()

            home_path = Models.get_ref_model_path(
                dummy_args,
                model.__class__.__name__,
                D164.name,
                suffix_str=tag + "." + model.netid,
            )
            model_path = os.path.join(home_path, "model.best.pth")
        else:
            model = Global.dataset_reference_ALI[args.dataset][0]()
            home_path = Models.get_ref_model_path(
                dummy_args,
                model.__class__.__name__,
                D164.name,
                suffix_str=tag + "." + model.netid,
            )
            model_path = os.path.join(home_path, "model.best.pth")

        model.load_state_dict(torch.load(model_path))

        model = model.to("cuda")

        d2s = []
        for y, d2 in enumerate(All_OD1):
            dataset = Global.all_datasets[d2]
            if "dataset_path" in dataset.__dict__:
                print(os.path.join(args.root_path, dataset.dataset_path))
                D2 = dataset(
                    root_path=os.path.join(args.root_path, dataset.dataset_path)
                ).get_D2_test(D164)

            else:
                D2 = dataset().get_D2_test(D164)
            d2s.append(D2)

        ALL_EMBS, ALL_Y, Cat2Y = proc_data(args, model, D1, d2s, All_OD1)

        with open(
            os.path.join(
                args.experiment_path,
                "cat2y_UC1_ppd_%d_d1_%s.pkl" % (args.points_per_d2, args.dataset),
            ),
            "wb",
        ) as fp:
            _pickle.dump(Cat2Y, fp)
        np.save(
            os.path.join(
                args.experiment_path,
                "all_y_UC1_ppd_%d_d1_%s.npy" % (args.points_per_d2, args.dataset),
            ),
            ALL_Y,
        )
        np.save(
            os.path.join(
                args.experiment_path,
                "all_embs_UC1_ppd_%d_d1_%s.npy" % (args.points_per_d2, args.dataset),
            ),
            ALL_EMBS,
        )

        #######################################################################################
        d2s = []
        OD2 = ALL_OD2_NIH if args.dataset == "NIHCC" else ALL_OD2_PAD
        for y, d2 in enumerate(OD2):
            dataset = Global.all_datasets[d2]
            if "dataset_path" in dataset.__dict__:
                print(os.path.join(args.root_path, dataset.dataset_path))
                D2 = dataset(
                    root_path=os.path.join(args.root_path, dataset.dataset_path)
                ).get_D2_test(D164)

            else:
                D2 = dataset().get_D2_test(D164)
            d2s.append(D2)

        ALL_EMBS, ALL_Y, Cat2Y = proc_data(args, model, D1, d2s, OD2)

        with open(
            os.path.join(
                args.experiment_path,
                "cat2y_UC2_ppd_%d_d1_%s.pkl" % (args.points_per_d2, args.dataset),
            ),
            "wb",
        ) as fp:
            _pickle.dump(Cat2Y, fp)
        np.save(
            os.path.join(
                args.experiment_path,
                "all_y_UC2_ppd_%d_d1_%s.npy" % (args.points_per_d2, args.dataset),
            ),
            ALL_Y,
        )
        np.save(
            os.path.join(
                args.experiment_path,
                "all_embs_UC2_ppd_%d_d1_%s.npy" % (args.points_per_d2, args.dataset),
            ),
            ALL_EMBS,
        )

        #########################################################################################
        d2s = []
        d3_tags = d3_tags_NIH if args.dataset == "NIHCC" else d3_tags_PAD
        for d2 in d3_tags:
            if args.dataset == "NIHCC":
                D2 = NIHChest(
                    root_path=os.path.join(args.root_path, "NIHCC"),
                    binary=True,
                    test_length=5000,
                    keep_in_classes=[
                        d2,
                    ],
                ).get_D2_test(D164)
            elif args.dataset == "PADChest":
                D2 = PADChestSV(
                    root_path=os.path.join(args.root_path, "PADChest"),
                    binary=True,
                    test_length=5000,
                    keep_in_classes=[
                        d2,
                    ],
                    downsample=64,
                ).get_D2_test(D164)
            d2s.append(D2)
        ALL_EMBS, ALL_Y, Cat2Y = proc_data(args, model, D1, d2s, d3_tags)

        with open(
            os.path.join(
                args.experiment_path,
                "cat2y_UC3_ppd_%d_d1_%s.pkl" % (args.points_per_d2, args.dataset),
            ),
            "wb",
        ) as fp:
            _pickle.dump(Cat2Y, fp)
        np.save(
            os.path.join(
                args.experiment_path,
                "all_y_UC3_ppd_%d_d1_%s.npy" % (args.points_per_d2, args.dataset),
            ),
            ALL_Y,
        )
        np.save(
            os.path.join(
                args.experiment_path,
                "all_embs_UC3_ppd_%d_d1_%s.npy" % (args.points_per_d2, args.dataset),
            ),
            ALL_EMBS,
        )

    else:
        pass

    for i in range(3):
        uc_tag = i + 1
        ALL_EMBS = np.load(
            os.path.join(
                args.experiment_path,
                "all_embs_UC%i_ppd_%d_d1_%s.npy"
                % (uc_tag, args.points_per_d2, args.dataset),
            )
        )
        with open(
            os.path.join(
                args.experiment_path,
                "cat2y_UC%i_ppd_%d_d1_%s.pkl"
                % (uc_tag, args.points_per_d2, args.dataset),
            ),
            "rb",
        ) as fp:
            Cat2Y = _pickle.load(fp)
        ALL_Y = np.load(
            os.path.join(
                args.experiment_path,
                "all_y_UC%i_ppd_%d_d1_%s.npy"
                % (uc_tag, args.points_per_d2, args.dataset),
            )
        )
        N = ALL_EMBS.shape[0]
        ALL_EMBS = ALL_EMBS.reshape(N, -1)
        N_plot = int(args.plot_percent * ALL_EMBS.shape[0])
        rand_inds = np.arange(ALL_EMBS.shape[0])
        np.random.shuffle(rand_inds)
        rand_inds = rand_inds[:N_plot]
        ALL_Y = ALL_Y[rand_inds]
        if args.umap:
            tsne = umap.UMAP(
                n_neighbors=int(args.perplexity),
                min_dist=args.lr,
                n_components=2,
                metric="euclidean",
                n_epochs=args.n_iter,
            )
        else:
            tsne = TSNE(
                perplexity=args.perplexity, learning_rate=args.lr, n_iter=args.n_iter
            )
        palette = sns.color_palette("bright", 10)
        from matplotlib.colors import ListedColormap

        my_cmap = ListedColormap(palette.as_hex())

        X_embedded = tsne.fit_transform(ALL_EMBS)

        X_embedded = X_embedded[rand_inds]

        np.save(
            os.path.join(
                args.experiment_path,
                "embedded_UC%i_ppd_%d_d1_%s.npy"
                % (uc_tag, args.points_per_d2, args.dataset),
            ),
            X_embedded,
        )
        np.save(
            os.path.join(
                args.experiment_path,
                "selectedY_UC%i_ppd_%d_d1_%s.npy"
                % (uc_tag, args.points_per_d2, args.dataset),
            ),
            ALL_Y,
        )

        fig, ax = plt.subplots()
        for k, cla in Cat2Y.items():
            target_inds = np.nonzero(ALL_Y == cla)
            ax.scatter(
                X_embedded[target_inds, 0].squeeze(),
                X_embedded[target_inds, 1].squeeze(),
                c=palette.as_hex()[cla],
                label=k,
                s=3.0,
            )

        # ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=ALL_Y, cmap=my_cmap)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        # ax.legend([k for k in Cat2Y.keys()])
        # plt.show()
        if args.umap:
            plt.savefig(
                os.path.join(args.experiment_path, "UC_%i_umap.png" % uc_tag), dpi=200
            )
        else:
            plt.savefig(
                os.path.join(args.experiment_path, "UC_%i_tsne.png" % uc_tag), dpi=200
            )
        # sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=ALL_Y, legend='full', palette=palette)
        print("done")
