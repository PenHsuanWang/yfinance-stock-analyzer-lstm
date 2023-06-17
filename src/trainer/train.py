#%%
#
from abc import ABC, abstractmethod

import torch


class Trainer(ABC):

    def __init__(self, model):
        self._model = model

    def run_training_loop(self, epochs: int) -> None:
        raise NotImplementedError


class PytorchTrainer(Trainer):

    def __init__(self, model, criterion, optimizer, device,
                 training_data: torch.Tensor,
                 training_labels: torch.Tensor
                 ):
        super().__init__(model)
        self._criterion = criterion
        self._optimizer = optimizer
        self._device = device
        self._training_data = training_data
        self._training_labels = training_labels

    def run_training_loop(self, epochs: int) -> None:
        print(f"Training the model for {epochs} epochs")
        self._model.train()
        for epoch in range(epochs):
            self._model.train()
            outputs = self._model(self._training_data)
            loss = self._criterion(outputs, self._training_labels)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item()}")
        print("Training done")

