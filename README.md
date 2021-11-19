# M2CS Pytorch

## Datasets
- `test_data`: The data for testing.
  - `CUHK`: Contains 100 testing images of CUHK Dataset and it's GT, click here. [CUHK](https://github.com/jerysaw/M2CS/releases/download/model/CUHK.rar)
  - `DUT`: Contains 500 testing images of DUT Dataset and it's GT. [DUT](https://github.com/jerysaw/M2CS/releases/download/model/DUT.rar)

### Test
You can use the following command to test：
> python test.py --stict PRETRAINED_WEIGHT --image_path --image_path IMG_PATH --mask_save_path SAVE_PATH

We provide a [pre-trained model](https://github.com/jerysaw/M2CS/releases/download/model/generator_pretrained.rar) for testing.
