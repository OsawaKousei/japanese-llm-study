from typing import List, Tuple

import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from transformer_layer import Transformer


class Trainer:
    def __init__(
        self,
        net: nn.Module,
        optimizer: optim.Optimizer,
        critetion: nn.Module,
        device: torch.device,
    ) -> None:
        self.net = net
        self.optimizer = optimizer
        self.critetion = critetion
        self.device = device
        self.net = self.net.to(self.device)

    def loss_fn(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.critetion(preds, labels)

    def train_step(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.net.train()
        output = self.net(src, tgt)

        tgt = tgt[
            :, 1:
        ]  # decoderからの出力は1 ~ max_lenまでなので、0以降のデータで誤差関数を計算する
        output = output[:, :-1, :]  #

        # calculate loss
        loss = self.loss_fn(
            output.contiguous().view(
                -1,
                output.size(-1),
            ),
            tgt.contiguous().view(-1),
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, output

    def val_step(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.net.eval()
        output = self.net(src, tgt)

        tgt = tgt[:, 1:]
        output = output[:, :-1, :]  #

        loss = self.loss_fn(
            output.contiguous().view(
                -1,
                output.size(-1),
            ),
            tgt.contiguous().view(-1),
        )

        return loss, output

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, print_log: bool = True
    ) -> Tuple[List[float], List[float], List[float]]:
        # train
        train_losses: List[float] = []
        if print_log:
            print(f"{'-'*20 + 'Train' + '-'*20} \n")
        for i, (src, tgt) in enumerate(train_loader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            (
                loss,
                _,
            ) = self.train_step(src, tgt)
            src = src.to("cpu")
            tgt = tgt.to("cpu")

            if print_log:
                print(
                    f"train loss: {loss.item()}" + f"iter: {i+1}/{len(train_loader)} \n"
                )

            train_losses.append(loss.item())

        # validation
        val_losses: List[float] = []
        if print_log:
            print(f"{'-'*20 + 'Validation' + '-'*20} \n")
        for i, (src, tgt) in enumerate(val_loader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            loss, _ = self.val_step(src, tgt)
            src = src.to("cpu")
            tgt = tgt.to("cpu")

            if print_log:
                print(f"train loss: {loss.item()}, iter: {i+1}/{len(val_loader)} \n")

            val_losses.append(loss.item())

        return train_losses, train_bleu_scores, val_losses

    def test(self, test_data_loader: DataLoader) -> Tuple[List[float]]:
        test_losses: List[float] = []
        for _, (src, tgt) in enumerate(test_data_loader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            loss, _ = trainer.val_step(src, tgt)
            src = src.to("cpu")
            tgt = tgt.to("cpu")

            test_losses.append(loss.item())

        return test_losses


if __name__ == "__main__":
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set hyperparameters
    src_vocab_size = 100
    tgt_vocab_size = 100
    max_len = 100
    d_model = 512
    heads_num = 8
    d_ff = 2048
    N = 6
    dropout_rate = 0.1
    layer_norm_eps = 1e-5
    pad_idx = 0
    device = torch.device("cpu")

    # set model
    net = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        max_len,
        d_model,
        heads_num,
        d_ff,
        N,
        dropout_rate,
        layer_norm_eps,
        pad_idx,
        device,
    )

    # set optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    # set criterion
    criterion = nn.CrossEntropyLoss()

    # set trainer
    trainer = Trainer(net, optimizer, criterion, device)

    train_loader = DataLoader()
    val_loader = DataLoader()
    test_loader = DataLoader()

    # train
    train_losses, train_bleu_scores, val_losses, val_bleu_scores = trainer.fit(
        train_loader, val_loader
    )

    # test
    test_losses, test_bleu_scores = trainer.test(test_loader)

    # plot
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="validation loss")
    plt.plot(test_losses, label="test loss")
    plt.legend()
    plt.show()
    plt.savefig("loss.png")

    # save
    torch.save(net.state_dict(), "transformer.pth")
