import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint


def train_diffu(
    model,
    train_loader,
    val_loader,
    test_loader,
    n_gpus=1,
    max_epochs=1,
    every_n_epochs=None,
    **kwargs
):
    trainer = L.Trainer(
        default_root_dir="./rec_diffu",
        accelerator="auto",
        devices=n_gpus,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, every_n_epochs=every_n_epochs),
            LearningRateMonitor("epoch"),
        ],
        **kwargs,
    )
    trainer.logger._log_graph = (
        True  # If True, we plot the computation graph in tensorboard
    )
    trainer.logger._default_hp_metric = (
        None  # Optional logging argument that we don't need
    )

    trainer.fit(model, train_loader, val_loader)

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result
