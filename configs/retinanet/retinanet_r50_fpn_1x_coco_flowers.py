# The new config inherits a base config to highlight the necessary modification
_base_ = './retinanet_r50_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
        bbox_head=dict(num_classes=2))

# Modify dataset related settings
dataset_type = 'COCODataset'
data_root = 'data/aigriflower/'
classes = ('flower','Flowers',)
data = dict(
    train=dict(
        img_prefix= data_root +'train/',
        classes=classes,
        ann_file= data_root +'train/_annotations.coco.json'),
    val=dict(
        img_prefix= data_root + '/valid/',
        classes=classes,
        ann_file= data_root +'valid/_annotations.coco.json'),
    test=dict(
        img_prefix= data_root +'test/',
        classes=classes,
        ann_file= data_root +'test/_annotations.coco.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'