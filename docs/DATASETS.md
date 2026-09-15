# Dataset Preparation

Datasets are placed under `artifact/data/` by default. Use `--data-root /path/to/data` to select another dataset root.

## CIFAR-10

- **Official page:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Download:** automatic through torchvision when Claim 1, 3, or 5 is first run
- **Path:** `artifact/data/cifar-10-batches-py/`

No manual preparation is required. The corresponding experiment scripts enable
`--download-cifar10` and reuse the downloaded files on subsequent runs.

## CelebA

- **Official page:** [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **Download used:** [CelebA on Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

Extract the prepared Kaggle package into this layout:

```text
artifact/data/celeba/
|-- img_align_celeba/
|   `-- *.jpg
|-- list_attr_celeba.csv
`-- list_eval_partition.csv
```

`list_attr_celeba.csv` must contain `image_id`, `Smiling`, `Male`, and `Young`. `list_eval_partition.csv` must contain `image_id` and `partition`.

All JPG files must be placed directly under `img_align_celeba/`.

## NIH ChestXray14

- **Official page:** [NIH ChestXray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- **Download used:** [224x224 resized package on Kaggle](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized)

The prepared Kaggle package is used to avoid repeating image resizing. Arrange the files as follows:

```text
artifact/data/nih_chest_xray/
|-- images-224/
|   `-- *.png
|-- Data_Entry_2017.csv
|-- train_val_list_NIH.txt
`-- test_list_NIH.txt
```

All PNG files must be placed directly under `images-224/`.
