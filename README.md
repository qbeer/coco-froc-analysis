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
    110.72555541992188,
    13.9161834716797,
    49.4566650390625,
    36.65155029296875
  ]
}
```

The FROC analysis counts the number of images, number of lesions in the ground truth file for all categories and then counts the lesion localization predictions and the non-lesion localization predictions. A lesion is localized by default if its center is inside any ground truth box and the categories match or if you wish to use IoU you should provide threshold upon which you can define the 'close enough' relation.

## Usage

```python
from froc_analysis import generate_froc_curve, generate_bootstrap_curves

# For single FROC curve
generate_froc_curve(gt_ann='<path-to-your-ground-thruth-annotation-file>',
                    pr_ann='<path-to-Detectron2-or-mmdetection-prediction-file>',
                    use_iou=False, iou_thres=.5, n_sample_points=75,
                    plot_title='FROC', plot_output_path='froc.png')

# For bootstrapped curves
generate_bootstrap_curves(gt_ann='<path-to-your-ground-thruth-annotation-file>',
                          pr_ann='<path-to-Detectron2-or-mmdetection-prediction-file>',
                          n_bootstrap_samples=5,
                          use_iou=False, iou_thres=.5, n_sample_points=25,
                          plot_title='FROC', plot_output_path='froc.png')
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

### Running tests

```bash
python -m coverage run -m unittest discover --pattern "*_test.py" -v
python -m coverage report -m
```

### Building and publishing (reminder)

```bash
ace # for local CI pipeline
poetry publish --build -r testpypi # or without -r testpypi for publishing to pypi
```

@Regards, Alex
