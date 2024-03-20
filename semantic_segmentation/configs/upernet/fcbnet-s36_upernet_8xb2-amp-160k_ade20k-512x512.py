_base_ = [
    "../_base_/models/upernet_convnext.py",
    "../_base_/datasets/ade20k.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_160k.py",
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint_file = "/nfs/ruihanxu/fcbnet/fcbformer_s36.pth"
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type="MetaFormer",
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint_file),
    ),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=150,
    ),
    auxiliary_head=dict(in_channels=320, num_classes=150),
    test_cfg=dict(mode="slide", crop_size=crop_size, stride=(341, 341)),
)

optim_wrapper = dict(
    _delete_=True,
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg={"decay_rate": 0.9, "decay_type": "stage_wise", "num_layers": 6},
    constructor="MetaFormerLearningRateDecayOptimizerConstructor",
    loss_scale="dynamic",
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    ),
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
