# BDD100K SegFormer-B0 Evaluation Report

## Experiment summary

- Best epoch recorded during training: **24**
- Completed epoch in resume state: **56**
- Validation loss: **0.284399**
- mIoU, all 19 classes: **56.7721%**
- mIoU, classes with nonzero union: **56.7721%**
- Pixel accuracy: **92.5896%**
- Mean class accuracy: **65.8326%**
- Frequency-weighted IoU: **86.8523%**
- Validation images: **1000**

## Per-class metrics

|   class_id | class_name    | IoU_percent   | Dice_F1_percent   | Precision_percent   | Recall_percent   |   GT_pixels |   Predicted_pixels |   FP_pixels |   FN_pixels |
|-----------:|:--------------|:--------------|:------------------|:--------------------|:-----------------|------------:|-------------------:|------------:|------------:|
|          0 | road          | 94.48%        | 97.16%            | 96.97%              | 97.36%           |    49809765 |           50010998 |     1516020 |     1314787 |
|          1 | sidewalk      | 63.03%        | 77.32%            | 78.25%              | 76.42%           |     4726739 |            4616655 |     1004353 |     1114437 |
|          2 | building      | 84.09%        | 91.36%            | 90.24%              | 92.50%           |    34281843 |           35137532 |     3428477 |     2572788 |
|          3 | wall          | 33.67%        | 50.38%            | 53.00%              | 48.00%           |      821199 |             743739 |      349554 |      427014 |
|          4 | fence         | 45.72%        | 62.75%            | 66.15%              | 59.68%           |     1855443 |            1673851 |      566562 |      748154 |
|          5 | pole          | 41.50%        | 58.66%            | 69.41%              | 50.80%           |     2228308 |            1630824 |      498919 |     1096403 |
|          6 | traffic light | 42.85%        | 60.00%            | 71.09%              | 51.90%           |      311484 |             227412 |       65756 |      149828 |
|          7 | traffic sign  | 42.94%        | 60.08%            | 73.22%              | 50.94%           |      530816 |             369347 |       98923 |      260392 |
|          8 | vegetation    | 84.93%        | 91.85%            | 90.55%              | 93.19%           |    35507980 |           36545411 |     3454010 |     2416579 |
|          9 | terrain       | 47.23%        | 64.16%            | 68.30%              | 60.49%           |     2101954 |            1861423 |      589983 |      830514 |
|         10 | sky           | 95.18%        | 97.53%            | 98.00%              | 97.07%           |    41178228 |           40788262 |      816907 |     1206873 |
|         11 | person        | 58.67%        | 73.96%            | 81.49%              | 67.70%           |      632059 |             525108 |       97214 |      204165 |
|         12 | rider         | 44.62%        | 61.71%            | 73.85%              | 52.99%           |       22682 |              16274 |        4255 |       10663 |
|         13 | car           | 87.96%        | 93.60%            | 92.59%              | 94.62%           |    20908874 |           21366511 |     1582316 |     1124679 |
|         14 | truck         | 48.84%        | 65.62%            | 74.39%              | 58.71%           |     2337331 |            1844537 |      472401 |      965195 |
|         15 | bus           | 76.15%        | 86.46%            | 88.53%              | 84.49%           |     1449154 |            1383046 |      158699 |      224807 |
|         16 | train         | 0.00%         | 0.00%             | 0.00%               | 0.00%            |       27199 |               1360 |        1360 |       27199 |
|         17 | motorcycle    | 45.22%        | 62.28%            | 67.97%              | 57.46%           |       54780 |              46312 |       14834 |       23302 |
|         18 | bicycle       | 41.58%        | 58.74%            | 61.15%              | 56.51%           |       36366 |              33602 |       13053 |       15817 |

## Files

- `per_class_metrics.csv`: complete per-class metrics
- `confusion_matrix_counts.csv`: raw confusion matrix
- `confusion_matrix_row_normalized_percent.csv`: row-normalized confusion matrix
- `training_history.csv`: training history
- `training_loss.png`: train and validation loss plot, if history was available
- `validation_miou.png`: mIoU curve, if history was available
- `preprocessing_config.json`: input size, normalization, label IDs and augmentation
- `resume/last_checkpoint.pt`: latest saved model/optimizer/scheduler state, if available

## Interpretation notes

- IoU is intersection divided by union.
- Dice/F1 is the class-level Dice coefficient.
- Precision is TP/(TP+FP); recall is TP/(TP+FN).
- The confusion matrix uses rows as ground-truth classes and columns as predicted classes.
- `255` is the ignore index and is excluded from evaluation.
- The reported model is the saved best checkpoint, not necessarily the model weights from the last completed epoch.
