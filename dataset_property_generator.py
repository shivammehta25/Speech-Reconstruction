
from src.utilities.data_utils import SpeechCommandDataset, CustomCollate
from torchaudio.datasets.speechcommands import SPEECHCOMMANDS
from torch.utils.data.dataloader import DataLoader
from src.hparams import create_hparams
from tqdm import tqdm
import torch


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors = []

    # Gather in lists, and encode labels as indices
    for waveform, *_ in batch:
        tensors += [waveform]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)

    return tensors


def generate_mean_and_std_of_speechcommands():

    hparams = create_hparams()
    dataset = SpeechCommandDataset(hparams, subset='training', normalize=False)
    data_loader = DataLoader(dataset, hparams.batch_size,
                             collate_fn=CustomCollate(80))

    total_sum = 0.0
    total_len = 0
    total_sum_sq = 0.0

    max_value, min_value = -float("inf"), float("inf")

    for i, batch in enumerate(tqdm(data_loader)):
        mel, mel_len = batch
        mel = mel.cuda()
        mel_len = mel_len.cuda()
        total_sum += torch.sum(mel)
        total_len += torch.sum(mel_len)
        total_sum_sq += torch.sum(torch.pow(mel, 2))

        max_batch = torch.max(mel)
        min_batch = torch.min(mel)

        if max_batch > max_value:
            max_value = max_batch

        if min_batch < min_value:
            min_value = min_batch

    mean = total_sum / (total_len * hparams.n_mel_channels)

    mean_sq = total_sum_sq / (total_len * hparams.n_mel_channels)
    variance = mean_sq - (mean * mean)
    std = torch.sqrt(variance)
    print("Mean and variance ", mean, std)
    print("Max and Min", max_value, min_value)

    # torch.save({'mean': mean, 'std': std}, "data_mean_and_std.pt")
    torch.save({'mean': mean, 'std': std, 'min': min_value,
               'max': max_value}, "data_mean_and_std.pt")


if __name__ == "__main__":
    generate_mean_and_std_of_speechcommands()
