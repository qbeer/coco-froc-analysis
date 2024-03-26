# COCO FROC analysis

FROC analysis for COCO annotations and Detectron(2) detection results. The COCO annotation style is defined [here](https://cocodataset.org/).

### Installation

```bash
pip install coco-froc-analysis
```

### About

A single annotation record in the ground-truth file might look like this:

```json
{
  "area": 2120,
  "iscrowd": 0,
  "bbox": [111, 24, 53, 40],
  "category_id": 3,
  "ignore": 0,
  "segmentation": [],
  "image_id": 407,
  "id": 945
}
```

While the prediction (here for bounding box) given by the region detection framework is such:

```json
{
  "image_id": 407,
  "category_id": 3,
  "score": 0.9990422129631042,
  "bbox": [
    110.72555541992188, 13.9161834716797, 49.4566650390625, 36.65155029296875
  ]
}
```

The FROC analysis counts the number of images, number of lesions in the ground truth file for all categories and then counts the lesion localization predictions and the non-lesion localization predictions. A lesion is localized by default if its center is inside any ground truth box and the categories match or if you wish to use IoU you should provide threshold upon which you can define the 'close enough' relation.

## Usage

```python
from coco_froc_analysis.count import generate_bootstrap_count_curves
from coco_froc_analysis.count import generate_count_curve
from coco_froc_analysis.froc import generate_bootstrap_froc_curves
from coco_froc_analysis.froc import generate_froc_curve

# For single FROC curve
generate_froc_curve(
            gt_ann=args.gt_ann,
            pr_ann=args.pr_ann,
            use_iou=args.use_iou,
            iou_thres=args.iou_thres,
            n_sample_points=args.n_sample_points,
            plot_title='FROC' if args.plot_title is None else args.plot_title,
            plot_output_path='froc.png' if args.plot_output_path is None else args.plot_output_path,
            test_ann=args.test_ann,
        )

# For bootstrapped curves
generate_bootstrap_froc_curves(
            gt_ann=args.gt_ann,
            pr_ann=args.pr_ann,
            n_bootstrap_samples=args.bootstrap,
            use_iou=args.use_iou,
            iou_thres=args.iou_thres,
            n_sample_points=args.n_sample_points,
            plot_title='FROC (bootstrap)' if args.plot_title is None else args.plot_title,
            plot_output_path='froc_bootstrap.png' if args.plot_output_path is None else args.plot_output_path,
            test_ann=args.test_ann,
        )
```

Please check `run.py` for more details. The `IoU` part of this code is not reliable and currently the codebase only works for binary evaluation, but any multiclass problem could be chunked up to work with it.

Description of `run.py` arguments:

```bash
usage: run.py [-h] [--bootstrap BOOTSTRAP] --gt_ann GT_ANN --pr_ann PR_ANN [--use_iou] [--iou_thres IOU_THRES] [--n_sample_points N_SAMPLE_POINTS]
              [--plot_title PLOT_TITLE] [--plot_output_path PLOT_OUTPUT_PATH] [--test_ann TEST_ANN] [--counts] [--weighted]

optional arguments:
  -h, --help            show this help message and exit
  --bootstrap BOOTSTRAP
                        Whether to do a single or bootstrap runs.
  --gt_ann GT_ANN
  --pr_ann PR_ANN
  --use_iou             Use IoU score to decide based on `proximity`
  --iou_thres IOU_THRES
                        If IoU score is used the default threshold is set to .5
  --n_sample_points N_SAMPLE_POINTS
                        Number of points to evaluate the FROC curve at.
  --plot_title PLOT_TITLE
  --plot_output_path PLOT_OUTPUT_PATH
  --test_ann TEST_ANN   Extra ground-truth like annotations
  --counts
  --weighted
```

## CLI Usage

```bash
python -m coco_froc_analysis [-h] [--bootstrap N_BOOTSTRAP_ROUNDS] --gt_ann GT_ANN --pred_ann PRED_ANN [--use_iou] [--iou_thres IOU_THRES] [--n_sample_points N_SAMPLE_POINTS]
                        [--plot_title PLOT_TITLE] [--plot_output_path PLOT_OUTPUT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --bootstrap  N_ROUNDS Whether to do a single or bootstrap runs.
  --gt_ann GT_ANN
  --pred_ann PRED_ANN
  --use_iou             Use IoU score to decide on `proximity` rather then using center pixel inside GT box.
  --iou_thres IOU_THRES
                        If IoU score is used the default threshold is arbitrarily set to .5
  --n_sample_points N_SAMPLE_POINTS
                        Number of points to evaluate the FROC curve at.
  --plot_title PLOT_TITLE
  --plot_output_path PLOT_OUTPUT_PATH
```

By default centroid closeness is used, if the `--use_iou` flag is set, `--iou_thres` defaults to `.75` while the `--score_thres` score defaults to `.5`. The code outputs the FROC curve on the given detection results and GT dataset.

## For developers

### Installing

In order to develop this repository you are in need of `poetry`. To install the latest version of `poetry` please run:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

After that you can install the dependencies by running:

```bash
poetry install
```

This will install the dependencies and the package in the virtual environment that you can activate by running:

```bash
poetry shell
```

In order to have checks ready before pushing you should also install `pre-commit`:

```bash
pre-commit install
```

### Running tests

```bash
python -m coverage run -m unittest discover --pattern "*_test.py" -v
python -m coverage report -m
```

### Testing GitHub Actions locally

Install `act` following the instructions [here](https://nektosact.com/installation/index.html).

```bash
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```

Then you can run the GitHub Actions locally by running:

```bash
act
```

This will always fail at the last step because the repository needs the `CODECOV_TOKEN` to be set in the repository secrets. This is not present locally, but this is not an ERROR.

### Creating documentation

```bash
pdoc -d google coco_froc_analysis -o docs # build docs
```

@Regards, Alex

```
@misc{qbeer,
  author       = {Alex Olar},
  title        = {FROC analysis for COCO-like file format},
  howpublished = {GitHub repository},
  month        = {September},
  year         = {2022},
  url          = {https://github.com/qbeer/coco-froc-analysis}
}
```
