from copy import deepcopy
import json
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import random
import os
import numpy as np

from .froc_curve import generate_froc_curve, froc_point, calc_scores
from .utils import transform_gt_into_pr


def generate_bootstrap_froc_curves(gt_ann,
                                   pr_ann,
                                   n_bootstrap_samples=5,
                                   use_iou=False,
                                   iou_thres=0.5,
                                   n_sample_points=50,
                                   plot_title="Bootstrap FROC",
                                   plot_output_path="froc_bootstrapped.png",
                                   test_ann=None):
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
            for ind, selected_image in enumerate(selected_images):
                if selected_image["id"] == img_id:
                    new_gt_ann = deepcopy(_gt_ann_)
                    new_gt_ann["image_id"] = ind
                    bootstrap_gt["annotations"].append(new_gt_ann)

        predictions = []

        for ind, img in enumerate(selected_images):
            for pr in PRED_ANN:
                if pr["image_id"] == img["id"]:
                    new_pr = deepcopy(pr)
                    new_pr["image_id"] = ind
                    predictions.append(new_pr)

        re_indexed_images = []
        for ind in range(len(selected_images)):
            image = deepcopy(selected_images[ind])
            image["id"] = ind
            re_indexed_images.append(image)

        bootstrap_gt["images"] = re_indexed_images

        with open("/tmp/tmp_bootstrap_gt.json", "w") as fp:
            json.dump(bootstrap_gt, fp)

        with open("/tmp/tmp_bootstrap_pred.json", "w") as fp:
            json.dump(predictions, fp)

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
            collected_frocs["lls"][cat_id].append(lls[cat_id])
            collected_frocs["nlls"][cat_id].append(nlls[cat_id])

    interpolated_frocs = {}
    max_froc_lls = {}
    min_froc_lls = {}

    for cat_id in collected_frocs["lls"]:
        all_lls = np.array(collected_frocs["lls"][cat_id]).reshape(
            n_bootstrap_samples, n_sample_points)
        all_nlls = np.array(collected_frocs["nlls"][cat_id]).reshape(
            n_bootstrap_samples, n_sample_points)

        _, max_nlls = np.min(all_nlls), np.max(all_nlls)

        x_range = np.linspace(1e-2, max_nlls, n_sample_points, endpoint=True)

        frocs = []

        for lls, nlls in zip(all_lls, all_nlls):
            interpolated_lls = np.interp(x_range, nlls[::-1], lls[::-1])
            frocs.append(interpolated_lls)

        interpolated_frocs[cat_id] = {
            "nlls": x_range,
            "lls": np.array(frocs).reshape(n_bootstrap_samples,
                                           n_sample_points)
        }

        max_froc_lls[cat_id] = np.max(
            np.array(frocs).reshape(n_bootstrap_samples, n_sample_points),
            axis=0,
        )

        min_froc_lls[cat_id] = np.min(
            np.array(frocs).reshape(n_bootstrap_samples, n_sample_points),
            axis=0,
        )

    mean_froc_curve = {}

    for cat_id in interpolated_frocs:
        mean_froc_curve[cat_id] = np.stack(
            (interpolated_frocs[cat_id]['nlls'],
             np.mean(interpolated_frocs[cat_id]['lls'], axis=0)),
            axis=-1)

        plt.semilogx(
            mean_froc_curve[cat_id][:, 0],
            mean_froc_curve[cat_id][:, 1],
            "b-",
            label="mean",
        )

        plt.fill_between(interpolated_frocs[cat_id]['nlls'],
                         min_froc_lls[cat_id],
                         max_froc_lls[cat_id],
                         alpha=.2)

        if test_ann is not None:
            for t_ann in test_ann:
                t_pr = transform_gt_into_pr(t_ann, gt_ann)
                stats = froc_point(gt_ann, t_pr, .5, use_iou, iou_thres)
                _lls_accuracy, _nlls_per_image = calc_scores(stats, {}, {})
                label = t_ann.split('/')[-1].replace('.json', '')
                plt.plot(_nlls_per_image[cat_id][0],
                         _lls_accuracy[cat_id][0],
                         '+',
                         markersize=12,
                         label=label)

    plt.xlabel("FP/image")
    plt.ylabel("Sensitivity")

    plt.legend(loc="upper left")

    os.remove("/tmp/tmp_bootstrap_gt.json")
    os.remove("/tmp/tmp_bootstrap_pred.json")

    plt.title(plot_title)

    plt.savefig(plot_output_path, dpi=100)
