import wandb
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

dataset = MNIST(root="./MNIST", download=True, transform=transform)
training_set, validation_set = random_split(dataset, [55000, 5000])

training_loader = DataLoader(training_set, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=64)

import torch
from pytorch_lightning import LightningModule
from torch.nn import CrossEntropyLoss, Linear
from torch.nn import functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy


class MNIST_LitModule(LightningModule):
    def __init__(self, n_classes=10, n_layer_1=128, n_layer_2=256, lr=1e-3):
        """method used to define our model parameters"""
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = Linear(28 * 28, n_layer_1)
        self.layer_2 = Linear(n_layer_1, n_layer_2)
        self.layer_3 = Linear(n_layer_2, n_classes)

        # loss
        self.loss = CrossEntropyLoss()

        # optimizer parameters
        self.lr = lr

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    def forward(self, x):
        """method used for inference input -> output"""

        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # let's do 3 x (linear + relu)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        return x

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch"""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)

        # Let's return preds to use it in a custom callback
        return preds

    def test_step(self, batch, batch_idx):
        """used for logging metrics"""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("test_loss", loss)
        self.log("test_accuracy", acc)

    def configure_optimizers(self):
        """defines model optimizer"""
        return Adam(self.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar"""
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        acc = accuracy(preds, y, "multiclass", num_classes=10)
        return preds, loss, acc


model = MNIST_LitModule(n_layer_1=128, n_layer_2=128)

from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(
    project="MNIST", log_model="all"  # group runs in "MNIST" project
)  # log all new checkpoints during training


# from pytorch_lightning.callbacks import Callback


# class LogPredictionsCallback(Callback):
#     def on_validation_batch_end(
#         self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
#     ):
#         """Called when the validation batch ends."""

#         # `outputs` comes from `LightningModule.validation_step`
#         # which corresponds to our model predictions in this case

#         # Let's log 20 sample image predictions from first batch
#         if batch_idx == 0:
#             n = 20
#             x, y = batch
#             images = [img for img in x[:n]]
#             captions = [
#                 f"Ground Truth: {y_i} - Prediction: {y_pred}"
#                 for y_i, y_pred in zip(y[:n], outputs[:n])
#             ]

#             # Option 1: log images with `WandbLogger.log_image`
#             wandb_logger.log_image(key="sample_images", images=images, caption=captions)

#             # Option 2: log predictions as a Table
#             columns = ["image", "ground truth", "prediction"]
#             data = [
#                 [wandb.Image(x_i), y_i, y_pred]
#                 for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))
#             ]
#             wandb_logger.log_table(key="sample_table", columns=columns, data=data)


# log_predictions_callback = LogPredictionsCallback()

trainer = Trainer(
    logger=wandb_logger,  # W&B integration
    callbacks=[
        # log_predictions_callback,  # logging of sample predictions
        checkpoint_callback,
    ],  # our model checkpoint callback
    accelerator="gpu",  # use GPU
    max_epochs=5,
)  # number of epochs

trainer.fit(model, training_loader, validation_loader)
