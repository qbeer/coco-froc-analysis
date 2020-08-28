# COCO FROC analysis

FROC analysis for COCO annotations and Detectron2 results. The COCO annotation style is defined [here]().

### Example

A single annotation record in the file might look like this:

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

No dependencies.

```bash
python froc.py --gt_ann <path_to_ground_truth_annotation_in_COCO_format>\
               --pred_ann <path_to_prediction_annotation_in_COCO_format>\
               --use_iou <flag_parameter_if_used_then_it_is_automatically_set_to_true>\
               --iou_thres <can_be_used_with_the_above_optional_flag>\
               --confidence <confidence_of_the_predictions_above_which_detections_will_be_considered>
# arguments are required: --gt_ann, --pred_ann
```

By default centroid closeness is used, if the `--use_iou` flag is set, `--iou_thres` defaults to `.75` while the `--confidence` score defaults to `.5`. The code outputs statistics in the dictionary of form:

```python
{
    1 : # category id
    {'LL': ..., # lesion-localization count
     'NL': ..., # non-lesion localization count
     'n_images': ...,
     'n_lesions': ...,
     'name': ...}, # category name
...
}
```

By setting the confidence of the predictor you can generate points on the FROC curve.

## @Regards, Alex
