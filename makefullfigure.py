"""
First run tsne_encoder.py until the visualizations look good, and then set tsne_cache path to that experiment dir.
"""


import time
import numpy as np
import os
import matplotlib
import mlflow

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import os
import _pickle
from process_results_multiple import weighted_std
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import skimage.io as skio
import skimage.transform as skts


from matplotlib.colors import ListedColormap


def retry(f):
    while True:
        try:
            return f()
        except:
            time.sleep(1)
            print("Retrying mlflow.")


def setup_mlflow():
    os.environ["MLFLOW_TRACKING_USERNAME"] = "exp-01.mlflow-yang.ericst"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "parolaeric"
    remote_server_uri = "https://exp-01.mlflow-yang.inf.ethz.ch"
    retry(lambda: mlflow.set_tracking_uri(remote_server_uri))


def plot_tsne(exp_path, uc, ax, color_space=None, n_color_max=20, cat2color=None):
    X_embedded = None
    ALL_Y = None
    Cat2Y = None
    for (dirpath, dirnames, filenames) in os.walk(exp_path):
        for filename in filenames:
            if ("embedded_UC%i" % uc) in filename:
                X_embedded = np.load(os.path.join(exp_path, filename))
            elif ("selectedY_UC%i" % uc) in filename:
                ALL_Y = np.load(os.path.join(exp_path, filename))
            elif ("cat2y_UC%i" % uc) in filename:
                with open(os.path.join(exp_path, filename), "rb") as fp:
                    Cat2Y = _pickle.load(fp)
    if any([X_embedded is None, ALL_Y is None, Cat2Y is None]):
        raise FileNotFoundError("Missing some files from exp dir: %s" % exp_path)
    else:
        Y2Cat = dict([(v, k) for k, v in Cat2Y.items()])
        cat_labels = [Y2Cat[y] for y in ALL_Y]

        if color_space is None:
            assert cat2color is None
            color_space = sns.color_palette("hls", n_color_max)
            cat2color = {}
            i = 0
            for y, cat in Y2Cat.items():
                cat2color[cat] = color_space[i]
                i += 1
            this_color_list = [c for cat, c in cat2color.items()]
            this_cmap = ListedColormap(this_color_list)
        else:
            assert cat2color is not None
            n_used_colors = len(list(cat2color.values()))
            this_color_list = []
            for y, cat in Y2Cat.items():
                if cat in cat2color.keys():
                    this_color_list.append(cat2color[cat])
                else:
                    # new category
                    this_color_list.append(color_space[n_used_colors])
                    cat2color[cat] = color_space[n_used_colors]
                    n_used_colors += 1
            this_cmap = ListedColormap(this_color_list)

        df = pd.DataFrame(
            {"x": X_embedded[:, 0], "y": X_embedded[:, 1], "Dataset": cat_labels}
        )
        sns.scatterplot(
            "x",
            "y",
            hue="Dataset",
            s=1.0,
            data=df,
            ax=ax,
            linewidth=0,
            palette=cat2color,
        )
        ax.set_axis_off()
        ax.legend_.remove()
        # ax.legend(loc='upper right')
    return ax, color_space, cat2color


def plot_image(file_path, ax, uc):
    img = skio.imread(file_path)
    long_edge = max(img.shape[0], img.shape[1])
    img = skts.resize(img, (long_edge, long_edge))
    if len(img.shape) > 2:
        ax.imshow(img)
    else:
        ax.imshow(img, cmap="gray")
    ax.set_axis_off()
    ax.annotate(
        "Usecase %d" % uc,
        xy=(0.0, 0.5),
        xycoords="axes fraction",
        fontsize=5,
        xytext=(-5, 12),
        textcoords="offset points",
        ha="center",
        va="baseline",
        rotation=90,
    )
    # ax.set_ylabel("Usecase %d" % uc)
    return ax


def load_reto(d1, uc, pretrained: bool = False):
    if d1 == "NIHCC":
        D_id = "nih_id"
        D_ood = {
            1: "uc1_and_mura",
            2: "pc_for_nih",
            3: "nih_ood",
        }[uc]
    elif d1 == "PAD":
        D_id = "pc_id"
        D_ood = {
            1: "uc1_and_mura",
            2: "pc_uc2",
            3: "pc_uc3",
        }[uc]
    elif d1 == "DRD":
        D_id = "drd"
        D_ood = {
            1: "uc1_rgb",
            2: "drimdb",
            3: "riga",
        }[uc]
    else:
        assert False, f"Unknown dataset: {d1}"

    exp_name = f"{D_id}_vs_{D_ood}"

    exp = retry(lambda: mlflow.get_experiment_by_name(exp_name))
    runs = retry(lambda: mlflow.list_run_infos(exp.experiment_id))

    stats = {
        "auroc": [],
        "aupr": [],
    }

    for run_info in runs:
        run = retry(lambda: mlflow.get_run(run_info.run_id))
        if run.data.params.get("ensemble_type") != "assign_one_label":
            continue

        if run.data.params.get("use_pretrained_model") != str(pretrained):
            continue

        if run.data.params.get("model_arch") != "densenet":
            continue

        if run.data.tags.get("goal") != "final":
            continue

        metrics = run.data.metrics

        for k, v in {
            "auroc": metrics["heur_auroc_avg_diff"],
            "aupr": metrics["heur_aupr_avg_diff"],
        }.items():
            stats[k].append(v)

    for k, v in stats.items():
        assert len(v) > 0, f"Could not find any stats for {k}."

    return {k: np.mean(v) for k, v in stats.items()}


def plot_result(
    exp_path,
    d1,
    uc,
    ax,
    order=None,
    c_order=None,
    with_x_tick=False,
    keep_only_handles=None,
    alias=None,
    color_scheme="rainbow",
    legend=False,
    sort_separate=True,
):
    csv_data = np.load(os.path.join(exp_path, "data_UC%d_%s.npy" % (uc, d1)))
    assert os.path.isfile(os.path.join(exp_path, "headers_UC%d_%s.pkl" % (uc, d1)))
    with open(os.path.join(exp_path, "headers_UC%d_%s.pkl" % (uc, d1)), "rb") as fp:
        csv_headers = _pickle.load(fp)

    csv_headers[0].append("RETO")
    csv_headers[0].append("RETO(pretrained)")
    method_handles = csv_headers[0]

    weights = csv_data[0]
    uc_roc = csv_data[2] * 100
    uc_prc = csv_data[3] * 100

    if len(uc_roc.shape) == 3:
        rocm = np.average(uc_roc, axis=(1, 2), weights=weights)
        prcm = np.average(uc_prc, axis=(1, 2), weights=weights)
        rocv = weighted_std(uc_roc, weights, axis=(1, 2))
        prcv = weighted_std(uc_prc, weights, axis=(1, 2))
    else:
        new_weights = np.zeros_like(uc_roc)
        for i in range((uc_roc.shape[0])):
            for j in range((uc_roc.shape[1])):
                for k in range((uc_roc.shape[2])):
                    n = int(weights[i, j, k, 0])
                    for l in range(n):
                        new_weights[i, j, k, l] = 1
        rocm = np.average(uc_roc, axis=(1, 2, 3), weights=new_weights)
        prcm = np.average(uc_prc, axis=(1, 2, 3), weights=new_weights)
        rocv = (
            weighted_std(uc_roc, new_weights, axis=(1, 2, 3))
            * 1.96
            / np.sqrt(new_weights.sum((1, 2, 3)))
        )
        prcv = (
            weighted_std(uc_prc, new_weights, axis=(1, 2, 3))
            * 1.96
            / np.sqrt(new_weights.sum((1, 2, 3)))
        )

    reto_data = load_reto(d1, uc, pretrained=False)

    rocm = np.append(rocm, reto_data["auroc"] * 100)
    rocv = np.append(rocv, 0.0)

    prcm = np.append(prcm, reto_data["aupr"] * 100)
    prcv = np.append(prcv, 0.0)

    pretrained_reto_data = load_reto(d1, uc, pretrained=True)
    rocm = np.append(rocm, pretrained_reto_data["auroc"] * 100)
    rocv = np.append(rocv, 0.0)

    prcm = np.append(prcm, pretrained_reto_data["aupr"] * 100)
    prcv = np.append(prcv, 0.0)

    if order is None:
        if sort_separate:
            group1_handles_inds = []
            group2_handles_inds = []
            for i, handle in enumerate(method_handles):
                if ("ae" in handle.lower()) or ("ali" in handle.lower()):
                    group2_handles_inds.append(i)
                else:
                    group1_handles_inds.append(i)
            group1_sum = rocm[group1_handles_inds] + prcm[group1_handles_inds]
            group2_sum = rocm[group2_handles_inds] + prcm[group2_handles_inds]
            sorted_inds_g1 = [group1_handles_inds[i] for i in np.argsort(group1_sum)]
            sorted_inds_g2 = [group2_handles_inds[i] for i in np.argsort(group2_sum)]
            assert type(sorted_inds_g1) is list
            full_inds = np.array(sorted_inds_g1 + sorted_inds_g2)
        else:
            group_sums = rocm  # + prcm
            sorted_inds_g1 = np.argsort(group_sums)
            # assert type(sorted_inds_g1) is list
            full_inds = np.array(sorted_inds_g1)
    else:
        if type(order) is dict:
            order = [order[m] for m in method_handles]
            full_inds = np.zeros(len(order), dtype=int)
            for i, j in enumerate(order):
                full_inds[j] = i
        else:
            full_inds = np.array(order)

    sorted_rocm = rocm[full_inds]
    sorted_rocv = rocv[full_inds]

    sorted_prcm = prcm[full_inds]
    sorted_prcv = prcv[full_inds]

    sorted_method_handles = [method_handles[i] for i in full_inds]

    def proc_var(m, v):
        upper = []
        lower = []
        for n in range(m.shape[0]):
            if m[n] - v[n] < 30.0:
                lower.append(m[n])
            else:
                lower.append(v[n])
            if m[n] + v[n] > 100.0:
                upper.append(100.0 - m[n])
            else:
                upper.append(v[n])
        return np.array([lower, upper])

    pp_rocv = proc_var(sorted_rocm, sorted_rocv)
    pp_prcv = proc_var(sorted_prcm, sorted_prcv)

    if keep_only_handles is not None:
        keep_inds = []
        for i, method in enumerate(sorted_method_handles):
            if method in keep_only_handles:
                keep_inds.append(i)
            else:
                print("leaving out %s" % method)
        keep_inds = np.array(keep_inds)
        sorted_rocm = sorted_rocm[keep_inds]
        sorted_prcm = sorted_prcm[keep_inds]
        pp_rocv = pp_rocv[:, keep_inds]
        pp_prcv = pp_prcv[:, keep_inds]
        sorted_method_handles = [sorted_method_handles[i] for i in keep_inds]
        full_inds = full_inds[keep_inds]

    ind = np.arange(len(sorted_rocm))  # the x locations for the groups
    width = 0.35  # the width of the bars

    group_inds0 = []
    group_inds1 = []
    group_inds2 = []
    if color_scheme == "rgb":
        sorted_colors = [(0.9, 0.0, 0.0, 1.0) for i in sorted_method_handles]
        sorted_colors_0 = [(0.1, 0.9, 0.1, 1.0) for i in sorted_method_handles]
        # sorted_colors_1 = [(0.0, 0.2, 0.8, 1.0) for i in sorted_method_handles]
    elif color_scheme == "bluered":
        sorted_colors = []
        for i, handle in enumerate(sorted_method_handles):
            if ("ae" in handle.lower()) or ("ali" in handle.lower()):
                sorted_colors.append((0.9, 0.1, 0.0, 1.0))
                group_inds1.append(i)
            elif handle == "knn/1" or handle == "knn/8":
                sorted_colors.append((0.0, 0.8, 0.2, 1.0))
                group_inds2.append(i)
            else:
                sorted_colors.append((0.1, 0.2, 0.9, 1.0))
                group_inds0.append(i)
        sorted_colors_0 = [
            (
                color[0],
                color[1],
                color[2],
                0.3,
            )
            for color in sorted_colors
        ]
        # sorted_colors_1 = [(color[0], color[1], color[2], 0.7,) for color in sorted_colors]
    else:
        rb = cm.get_cmap("rainbow")
        grad = np.linspace(0, 1, len(sorted_rocm))
        colors = [rb(g) for g in grad]
        if c_order is None:
            this_order = np.arange(0, len(full_inds))
            c_order = {}
            for c, i in zip(this_order, full_inds):
                c_order[i] = c
        else:
            this_order = [c_order[i] for i in full_inds]

        sorted_colors = [colors[i] for i in this_order]
        sorted_colors_0 = [
            (
                color[0],
                color[1],
                color[2],
                0.3,
            )
            for color in sorted_colors
        ]
        # sorted_colors_1 = [(color[0], color[1], color[2], 0.7,) for color in sorted_colors]
    sorted_colors = np.array(sorted_colors)
    sorted_colors_0 = np.array(sorted_colors_0)
    group_inds0 = np.array(group_inds0)
    group_inds1 = np.array(group_inds1)
    group_inds2 = np.array(group_inds2)

    rects10 = ax.bar(
        ind[group_inds0] - width * 0.5,
        sorted_rocm[group_inds0],
        width,
        # yerr=pp_rocv[:, group_inds0],
        label="AUROC",
        color=sorted_colors_0[group_inds0],
        error_kw={"elinewidth": 0.8},
    )
    rects11 = ax.bar(
        ind[group_inds1] - width * 0.5,
        sorted_rocm[group_inds1],
        width,
        # yerr=pp_rocv[:, group_inds1],
        label="AUROC",
        color=sorted_colors_0[group_inds1],
        error_kw={"elinewidth": 0.8},
    )
    rects12 = ax.bar(
        ind[group_inds2] - width * 0.5,
        sorted_rocm[group_inds2],
        width,
        # yerr=pp_rocv[:, group_inds2],
        label="AUROC",
        color=sorted_colors_0[group_inds2],
        error_kw={"elinewidth": 0.8},
    )
    # rects2 = ax.bar(ind, sorted_rocm, width, yerr=pp_rocv,
    #                label='AUROC', color=sorted_colors_0, error_kw={'elinewidth':0.8})
    rects30 = ax.bar(
        ind[group_inds0] + width * 0.5,
        sorted_prcm[group_inds0],
        width,
        # yerr=pp_prcv[:, group_inds0],
        label="AUPRC",
        color=sorted_colors[group_inds0],
        error_kw={"elinewidth": 0.8},
    )
    rects31 = ax.bar(
        ind[group_inds1] + width * 0.5,
        sorted_prcm[group_inds1],
        width,
        # yerr=pp_prcv[:, group_inds1],
        label="AUPRC",
        color=sorted_colors[group_inds1],
        error_kw={"elinewidth": 0.8},
    )
    rects32 = ax.bar(
        ind[group_inds2] + width * 0.5,
        sorted_prcm[group_inds2],
        width,
        # yerr=pp_prcv[:, group_inds2],
        label="AUPRC",
        color=sorted_colors[group_inds2],
        error_kw={"elinewidth": 0.8},
    )
    if with_x_tick:
        ax.set_xticks(ind)
        if alias is not None:
            this_method_handles = []
            for handle in sorted_method_handles:
                if handle in alias:
                    this_method_handles.append(alias[handle])
                else:
                    this_method_handles.append(handle)
        else:
            this_method_handles = sorted_method_handles
        ax.set_xticklabels(this_method_handles, rotation=-45, ha="left")
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
    ax.set_yticks(np.linspace(50, 100, 3))
    ax.set_ylim(30, 110)
    if legend:
        ax.legend(
            [rects12, rects32, rects10, rects30, rects11, rects31],
            [
                "AUROC, data-only",
                "AUPRC, data-only",
                "AUROC, classifier only",
                "AUPRC, classifier only",
                "AUROC, with auxilary NN",
                "AUPRC, with auxilary NN",
            ],
            loc="upper left",
            ncol=3,
            markerscale=0.6,
            labelspacing=0.3,  # default 0.5
            columnspacing=1.0,  # default 2.0
            borderaxespad=0.3,  # default 0.5
        )
    ax.axhline(y=50, linewidth=0.5, color=(0.3, 0.3, 0.35, 0.8), ls="--")
    ax.axhline(y=75, linewidth=0.5, color=(0.3, 0.3, 0.35, 0.8), ls="--")
    ax.axhline(y=100, linewidth=0.5, color=(0.3, 0.3, 0.35, 0.8), ls="--")
    # ax.legend()
    return ax, full_inds, c_order


if __name__ == "__main__":
    setup_mlflow()
    PREVIEW_DOUBLE = True
    parser = argparse.ArgumentParser()
    parser.add_argument("output_name", type=str, help="save images")
    parser.add_argument(
        "--dataset", type=str, default="NIHCC", help="PAD or NIHCC or DRD or PCAM"
    )

    args = parser.parse_args()
    SAVE_NAME = args.output_name  # "DRD_MAINFIG"#"PAD_MAINFIG" #"NIHCC_MAINFIG"
    if args.dataset == "NIHCC":
        TSNE_PATH = (
            "umap_nih_AEBCE"  # "umap_drd_AEBCE"#"umap_padchest_AEBCE" #"ALI_NIHCC"
        )
        BAR_PATH = "nih_proced_res_mode1"  # "drd_proced_res_2"#"pad_new_res" #"nih_res"
        D1 = "NIHCC"  # "DRD"#"PAD"#"NIHCC"
        IMG1 = "sample_images/mura.png"  # "sample_images/mnist.png"#"sample_images/tinyimagenet.JPEG"#"sample_images/mura.png"
        IMG2 = "sample_images/padchest_lateral.png"  # "sample_images/drimdb.png"#"sample_images/padchest_AP.png"#"sample_images/padchest_lateral.png"
        IMG3 = "sample_images/pneumothorax.png"  # "sample_images/riga.jpg"#"sample_images/pneumothorax.png"

        ORDER = {
            "score_svm/0": 0,
            "prob_threshold/0": 1,
            "odin/0": 2,
            "12Layer-AE-BCE": 3,
            "svknn": 4,
            "mseaeknn/1": 5,
            "bceaeknn/1": 6,
            "vaebceaeknn/1": 7,
            "bceaeknn/8": 8,
            "mseaeknn/8": 9,
            "12Layer-VAE-MSE": 10,
            "vaemseaeknn/1": 11,
            "12Layer-AE-MSE": 12,
            "vaebceaeknn/8": 13,
            "knn/1": 14,
            "ALI_reconst/0": 15,
            "vaemseaeknn/8": 16,
            "Maha1layer": 17,
            "binclass/0": 18,
            "12Layer-VAE-BCE": 19,
            "knn/8": 20,
            "Maha": 21,
            "RETO": 22,
            "RETO(pretrained)": 23,
        }

        N_MAX = 20
    elif args.dataset == "PAD":
        TSNE_PATH = "umap_padchest_AEBCE"
        BAR_PATH = "pad_proced_res_mode1"
        D1 = "PAD"
        IMG1 = "sample_images/tinyimagenet.JPEG"
        IMG2 = "sample_images/padchest_AP.png"
        IMG3 = "sample_images/pad_cardiomegaly.png"
        ORDER = {
            "bceaeknn/1": 0,
            "prob_threshold/0": 1,
            "odin/0": 2,
            "mseaeknn/1": 3,
            "bceaeknn/8": 4,
            "vaebceaeknn/1": 5,
            "score_svm/0": 6,
            "12Layer-AE-BCE": 7,
            "mseaeknn/8": 8,
            "12Layer-VAE-BCE": 9,
            "12Layer-AE-MSE": 10,
            "vaemseaeknn/1": 11,
            "knn/1": 12,
            "12Layer-VAE-MSE": 13,
            "vaebceaeknn/8": 14,
            "vaemseaeknn/8": 15,
            "svknn": 16,
            "Maha": 17,
            "knn/8": 18,
            "binclass/0": 19,
            "Maha1layer": 20,
            "RETO": 21,
            "RETO(pretrained)": 22,
        }
        N_MAX = 20
    elif args.dataset == "DRD":
        TSNE_PATH = "umap_drd_AEBCE"
        BAR_PATH = "drd_proced_res_mode1"
        D1 = "DRD"
        IMG1 = "sample_images/mnist.png"
        IMG2 = "sample_images/drimdb.png"
        IMG3 = "sample_images/riga_sq.jpg"
        ORDER = {
            "score_svm/0": 0,
            "prob_threshold/0": 1,
            "odin/0": 2,
            "svknn": 3,
            "vaemseaeknn/1": 4,
            "vaemseaeknn/8": 5,
            "vaebceaeknn/1": 6,
            "12Layer-AE-BCE": 7,
            "vaebceaeknn/8": 8,
            "Maha": 9,
            "Maha1layer": 10,
            "mseaeknn/1": 11,
            "binclass/0": 12,
            "12Layer-AE-MSE": 13,
            "12Layer-VAE-BCE": 14,
            "12Layer-VAE-MSE": 15,
            "knn/1": 16,
            "knn/8": 17,
            "bceaeknn/1": 18,
            "mseaeknn/8": 19,
            "bceaeknn/8": 20,
            "RETO": 21,
            "RETO(pretrained)": 22,
        }
        N_MAX = 10
    elif args.dataset == "PCAM":
        TSNE_PATH = "umap_pcam_AEBCE"
        BAR_PATH = "pcam_proced_res_mode1"
        D1 = "PCAM"
        IMG1 = "sample_images/malaria.png"
        IMG2 = "sample_images/IDC.png"
        ORDER = {
            "prob_threshold/0": 0,
            "svknn": 1,
            "odin/0": 2,
            "12Layer-AE-BCE": 3,
            "12Layer-AE-MSE": 4,
            "score_svm/0": 5,
            "mseaeknn/1": 6,
            "vaemseaeknn/1": 7,
            "vaemseaeknn/8": 8,
            "vaebceaeknn/8": 9,
            "vaebceaeknn/1": 10,
            "12Layer-VAE-BCE": 11,
            "mseaeknn/8": 12,
            "bceaeknn/1": 13,
            "bceaeknn/8": 14,
            "12Layer-VAE-MSE": 15,
            "knn/1": 16,
            "knn/8": 17,
            "binclass/0": 18,
            "Maha1layer": 19,
            "Maha": 20,
            "RETO": 21,
            "RETO(pretrained)": 22,
        }
        N_MAX = 15
    matplotlib.rc("axes", edgecolor=(0.3, 0.3, 0.3, 0.8))

    plt.rc("font", size=5)  # controls default text sizes
    plt.rc("axes", titlesize=5)  # fontsize of the axes title
    plt.rc("axes", labelsize=5)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=5)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=5)  # fontsize of the tick labels
    plt.rc("legend", fontsize=5)  # legend fontsize
    plt.rc("figure", titlesize=10)  # fontsize of the figure title

    kiohandles = [
        "prob_threshold/0",
        "score_svm/0",
        "binclass/0",
        "odin/0",
        "Maha",
        "Maha1layer",
        "svknn",
        "12Layer-AE-BCE",
        "12Layer-AE-MSE",
        "12Layer-VAE-BCE",
        "12Layer-VAE-MSE",
        "ALI_reconst/0",
        "knn/1",
        "knn/8",
        "bceaeknn/8",
        "vaebceaeknn/8",
        "mseaeknn/8",
        "vaemseaeknn/8",
        "bceaeknn/1",
        "vaebceaeknn/1",
        "mseaeknn/1",
        "vaemseaeknn/1",
        "RETO",
        "RETO(pretrained)",
    ]

    alias = {
        "prob_threshold/0": "Prob. threshold",
        "score_svm/0": "Score SVM",
        "binclass/0": "Binary classifier",
        "odin/0": "ODIN",
        "Maha": "Mahalanobis",
        "Maha1layer": "Single layer Maha.",
        "svknn": "Feature knn",
        "12Layer-AE-BCE": "Reconst. AEBCE",
        "12Layer-AE-MSE": "Reconst. AEMSE",
        "12Layer-VAE-MSE": "Reconst. VAEMSE",
        "12Layer-VAE-BCE": "Reconst. VAEBCE",
        "ALI_reconst/0": "Reconst. ALI",
        "knn/1": "KNN-1",
        "knn/8": "KNN-8",
        "bceaeknn/8": "AEBCE-KNN-8",
        "vaebceaeknn/8": "VAEBCE-KNN-8",
        "mseaeknn/8": "AEMSE-KNN-8",
        "vaemseaeknn/8": "VAEMSE-KNN-8",
        "bceaeknn/1": "AEBCE-KNN-1",
        "vaebceaeknn/1": "VAEBCE-KNN-1",
        "mseaeknn/1": "AEMSE-KNN-1",
        "vaemseaeknn/1": "VAEMSE-KNN-1",
        "RETO": "RETO",
        "RETO(pretrained)": "RETO(pretrained)",
    }

    catalias = {
        "UniformNoise": "Noise",
        "FashionMNIST": "Fashion",
        "PADChestAP": "PC Ant. Pos.",
        "PADChestL": "PC Lateral",
        "PADChestAPHorizontal": "PC AP Horizontal",
        "PADChestPED": "PC Pediatric",
        "RIGA": "Glaucoma",
    }

    if args.dataset == "PCAM":
        if PREVIEW_DOUBLE:
            fig = plt.figure(1, figsize=(6, 2.5), dpi=109 * 2)
        else:
            fig = plt.figure(
                1, figsize=(6, 2.5), dpi=109
            )  # dpi set to match preview to print size on a 14 inch tall, 1440p monitor.
            # set up subplot grid
        gridspec.GridSpec(2, 6)  # nrow by n col         # effectively 1 inch squres

        ax_L0 = plt.subplot2grid((2, 6), (0, 2), colspan=4, rowspan=1)
        # ax_L0.annotate(
        #     "C",
        #     xy=(0.0, 1),
        #     xycoords="axes fraction",
        #     fontsize=7,
        #     xytext=(0, 3),
        #     textcoords="offset points",
        #     ha="center",
        #     va="baseline",
        # )
        ax_L0.yaxis.tick_right()
        _, inds, corder = plot_result(
            BAR_PATH,
            D1,
            1,
            ax_L0,
            order=ORDER,
            keep_only_handles=kiohandles,
            color_scheme="bluered",
            sort_separate=False,
        )
        ax_L1 = plt.subplot2grid((2, 6), (1, 2), colspan=4, rowspan=1)
        ax_L1.yaxis.tick_right()
        plot_result(
            BAR_PATH,
            D1,
            2,
            ax_L1,
            inds,
            c_order=corder,
            keep_only_handles=kiohandles,
            color_scheme="bluered",
            with_x_tick=True,
            legend=True,
            alias=alias,
        )

        ax_s11 = plt.subplot2grid((2, 6), (0, 1), colspan=1, rowspan=1)
        ax_s11.annotate(
            "B",
            xy=(0.0, 1),
            xycoords="axes fraction",
            fontsize=7,
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="baseline",
        )
        _, color_space, cat2color = plot_tsne(TSNE_PATH, 1, ax_s11, n_color_max=N_MAX)
        ax_s21 = plt.subplot2grid((2, 6), (1, 1), colspan=1, rowspan=1)
        plot_tsne(
            TSNE_PATH,
            2,
            ax_s21,
            color_space=color_space,
            cat2color=cat2color,
            n_color_max=N_MAX,
        )

        cats = list(cat2color.keys())
        for i, cat in enumerate(cats):
            if cat in catalias:
                cats[i] = catalias[cat]
        colors = list(cat2color.values())
        id = cats.index("In-Data")
        cats = (
            [
                "In-data",
            ]
            + cats[:id]
            + cats[id + 1 :]
        )
        colors = (
            [
                colors[id],
            ]
            + colors[:id]
            + colors[id + 1 :]
        )
        markers = [
            plt.Line2D([0, 0], [0, 0], color=color, marker="o", linestyle="")
            for color in colors
        ]
        fig.legend(
            markers,
            cats,
            title="Visualization of AEBCE latent space",
            numpoints=1,
            loc="upper left",
            bbox_to_anchor=(0.0, 0.265),
            ncol=3,
            markerscale=0.6,
            labelspacing=0.2,  # default 0.5
            columnspacing=0.8,  # default 2.0
            borderaxespad=0.4,  # default 0.5
        )

        ax_s10 = plt.subplot2grid((2, 6), (0, 0), colspan=1, rowspan=1)
        ax_s10.annotate(
            "A",
            xy=(0.0, 1),
            xycoords="axes fraction",
            fontsize=7,
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="baseline",
        )
        plot_image(IMG1, ax_s10, uc=1)
        ax_s20 = plt.subplot2grid((2, 6), (1, 0), colspan=1, rowspan=1)
        plot_image(IMG2, ax_s20, uc=2)

        plt.subplots_adjust(
            wspace=0.1, hspace=0.1, bottom=0.265, top=0.95, right=0.95, left=0.018
        )  # bottom=0.205

        # plt.savefig(SAVE_NAME + ".svg")
        plt.savefig(SAVE_NAME + ".png")
        fig.show()
    else:
        if PREVIEW_DOUBLE:
            fig = plt.figure(1, figsize=(6, 3.5), dpi=109 * 2)
        else:
            fig = plt.figure(
                1, figsize=(6, 3.5), dpi=109
            )  # dpi set to match preview to print size on a 14 inch tall, 1440p monitor.
            # set up subplot grid
        gridspec.GridSpec(3, 6)  # nrow by n col         # effectively 1 inch squres
        ax_L0 = plt.subplot2grid((3, 6), (0, 0), colspan=6, rowspan=1)
        # ax_L0.annotate(
        #     "C",
        #     xy=(0.0, 1),
        #     xycoords="axes fraction",
        #     fontsize=7,
        #     xytext=(0, 3),
        #     textcoords="offset points",
        #     ha="center",
        #     va="baseline",
        # )
        ax_L0.yaxis.tick_right()
        _, inds, corder = plot_result(
            BAR_PATH,
            D1,
            1,
            ax_L0,
            order=ORDER,
            keep_only_handles=kiohandles,
            color_scheme="bluered",
            sort_separate=False,
        )
        ax_L1 = plt.subplot2grid((3, 6), (1, 0), colspan=6, rowspan=1)
        ax_L1.yaxis.tick_right()
        plot_result(
            BAR_PATH,
            D1,
            2,
            ax_L1,
            inds,
            c_order=corder,
            keep_only_handles=kiohandles,
            color_scheme="bluered",
            alias=alias,
        )

        ax_L2 = plt.subplot2grid((3, 6), (2, 0), colspan=6, rowspan=1)
        ax_L2.yaxis.tick_right()
        plot_result(
            BAR_PATH,
            D1,
            3,
            ax_L2,
            inds,
            c_order=corder,
            with_x_tick=True,
            keep_only_handles=kiohandles,
            alias=alias,
            color_scheme="bluered",
            legend=True,
        )

        plt.subplots_adjust(
            wspace=0.1, hspace=0.1, bottom=0.205, top=0.95, right=0.95, left=0.018
        )  # bottom=0.205

        # plt.savefig(SAVE_NAME + ".svg")
        plt.savefig(SAVE_NAME + ".png")
        fig.show()
    print("done")
