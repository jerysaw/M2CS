# M2CS Pytorch
pytorch+cuda11.2

## Datasets
- `test_data`: The data for testing.
  - `CUHK`: Contains 100 testing images of CUHK Dataset and it's GT. Download [here](https://github.com/jerysaw/M2CS/releases/download/model/CUHK.rar)
  - `DUT`: Contains 500 testing images of DUT Dataset and it's GT. Download [here](https://github.com/jerysaw/M2CS/releases/download/model/DUT.rar)

### Test
You can use the following command to test：
> python test.py --stict PRETRAINED_WEIGHT --image_path IMG_PATH --mask_save_path SAVE_PATH

For example:
> python test.py --stict generator_pretrained.pth --image_path CUHK/xu100-source/ --mask_save_path result

We provide a [pre-trained model](https://github.com/jerysaw/M2CS/releases/download/model/generator_pretrained.rar) for testing.

### Train
Since our article is being reviewed, it is not suitable to publish the training code now. If the paper is accepted, we promise to upload the training code at the first time.
