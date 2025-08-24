from deimkit import Trainer, Config, configure_dataset, configure_model

conf = Config.from_model_name("deim_hgnetv2_s")

# Optional
conf = configure_model(
    config=conf, 
    num_queries=100,   # Optional, default is 300
    pretrained=True,   # Optional, default is True
    freeze_at=-1       # Optional, default is -1 (no freezing)
)

# Required
conf = configure_dataset(
    config=conf,
    image_size=(640, 640),
    train_ann_file="/scratch/rmkyas002/coco_detection_data/train/_annotations.coco.json",
    train_img_folder="/scratch/rmkyas002/coco_detection_data/train",
    val_ann_file="/scratch/rmkyas002/coco_detection_data/valid/_annotations.coco.json",
    val_img_folder="/scratch/rmkyas002/coco_detection_data/valid",
    train_batch_size=16,
    val_batch_size=16,
    num_classes=2,
    output_dir="./outputs",
)

trainer = Trainer(conf)

# # Optional - Load from a previously trained checkpoint
# trainer.load_checkpoint("previous_best.pth")

# All arguments are optional, if not specified, the default values for the model will be used.
trainer.fit(
    epochs=200,          # Number of training epochs
    save_best_only=True, # Save only the best model checkpoint
    lr=0.0001,           # Learning rate
    lr_gamma=0.1,        # Learning rate annealing factor
    weight_decay=0.0001, # Weight decay
)