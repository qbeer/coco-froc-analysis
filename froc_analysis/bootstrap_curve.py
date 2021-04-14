from copy import deepcopy
import json
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import random
import os
import numpy as np

from .froc_curve import generate_froc_curve, froc_point, calc_scores
from .utils import transform_gt_into_pr


def generate_bootstrap_curves(
    gt_ann,
    pr_ann,
    n_bootstrap_samples=5,
    use_iou=False,
    iou_thres=0.5,
    n_sample_points=50,
    plot_title="Bootstrap FROC",
    plot_output_path="froc_bootstrapped.png",
    test_ann=None
):
    with open(gt_ann, "r") as fp:
        GT_ANN = json.load(fp)

    with open(pr_ann, "r") as fp:
        PRED_ANN = json.load(fp)

    n_images = len(GT_ANN["images"])

    plt.figure(figsize=(15, 15))

    collected_frocs = {"lls": {}, "nlls": {}}

    for _ in tqdm(range(n_bootstrap_samples)):
        selected_images = random.choices(GT_ANN["images"], k=n_images)
        bootstrap_gt = deepcopy(GT_ANN)

        del bootstrap_gt["images"]

        bootstrap_gt["images"] = selected_images

        gt_annotations = bootstrap_gt["annotations"]

        del bootstrap_gt["annotations"]

        bootstrap_gt["annotations"] = []
        for _gt_ann_ in gt_annotations:
            img_id = _gt_ann_["image_id"]
            for selected_image in selected_images:
                if selected_image["id"] == img_id:
                    bootstrap_gt["annotations"].append(_gt_ann_)

        with open("/tmp/tmp_bootstrap_gt.json", "w") as fp:
            json.dump(bootstrap_gt, fp)

        with open("/tmp/tmp_bootstrap_pred.json", "w") as fp:
            json.dump(PRED_ANN, fp)

        tmp_gt_ann = "/tmp/tmp_bootstrap_gt.json"
        tmp_pred_ann = "/tmp/tmp_bootstrap_pred.json"

        lls, nlls = generate_froc_curve(
            tmp_gt_ann,
            tmp_pred_ann,
            use_iou,
            iou_thres,
            n_sample_points,
            plot_title=None,
            plot_output_path=None,
        )

        for cat_id in lls:
            if collected_frocs["lls"].get(cat_id, None) is None:
                collected_frocs["lls"] = {cat_id: []}
            if collected_frocs["nlls"].get(cat_id, None) is None:
                collected_frocs["nlls"] = {cat_id: []}

        for cat_id in lls:
            plt.semilogx(nlls[cat_id], lls[cat_id], "b--", alpha=0.15)
            collected_frocs["lls"][cat_id].append(lls[cat_id])
            collected_frocs["nlls"][cat_id].append(nlls[cat_id])

    mean_froc_lls = {}
    mean_froc_nlls = {}

    for cat_id in collected_frocs["lls"]:
        mean_froc_lls[cat_id] = np.mean(
            np.array(collected_frocs["lls"][cat_id]).reshape(
                n_bootstrap_samples, n_sample_points
            ),
            axis=0,
        )
        mean_froc_nlls[cat_id] = np.mean(
            np.array(collected_frocs["nlls"][cat_id]).reshape(
                n_bootstrap_samples, n_sample_points
            ),
            axis=0,
        )

    mean_froc_curve = {}

    for cat_id in collected_frocs["lls"]:
        mean_froc_curve[cat_id] = np.stack(
            (mean_froc_nlls[cat_id], mean_froc_lls[cat_id]), axis=-1
        )
        plt.semilogx(
            mean_froc_curve[cat_id][:, 0],
            mean_froc_curve[cat_id][:, 1],
            "bx-",
            label="mean",
        )

        if test_ann is not None:
            for t_ann in test_ann:
                t_pr = transform_gt_into_pr(t_ann, gt_ann)
                stats = froc_point(gt_ann, t_pr, .5, use_iou, iou_thres)
                _lls_accuracy, _nlls_per_image = calc_scores(stats, {}, {})
                plt.plot(_nlls_per_image[cat_id][0], _lls_accuracy[cat_id][0],
                        '+', markersize=12, label=t_ann.split('/')[-3])

    plt.xlabel("FP/image")
    plt.ylabel("Sensitivity")

    plt.legend(loc="upper left")

    os.remove("/tmp/tmp_bootstrap_gt.json")
    os.remove("/tmp/tmp_bootstrap_pred.json")

    plt.title(plot_title)

    plt.savefig(plot_output_path, dpi=100)
