import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Sampler
from transformers import PreTrainedTokenizerBase
from ..data_util.formatter import BaseCodeChunkFormatter, BaseCodeToCompleteFormatter, PythonCommentCodeChunkFormatter, PythonCommentCodeToCompleteFormatter
from .dataset import FidTrainingDataset
import lightning as L


class FidDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[tuple[list[list[int]], list[int]]]) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]]:
        """
        Returns:
        - encoder_input_ids: torch.Tensor - Padded encoder input IDs.
        - encoder_attention_mask: torch.Tensor - Attention mask for the encoder.
        - decoder_input_ids: torch.Tensor - Padded decoder input IDs.
        - decoder_attention_mask: torch.Tensor - Attention mask for the decoder.
        - encoder_context_num: list[int] - Number of contexts for each sample.
        - encoder_context_lengths: list[list[int]] - Lengths of each context for each sample without padding.
        """
        # import pdb
        # pdb.set_trace()
        encoder_input_ids, decoder_input_ids = zip(*batch)

        encoder_context_num = [len(context_list) for context_list in encoder_input_ids]
        encoder_context_lengths = [
            len(context)
            for context_list in encoder_input_ids
            for context in context_list
        ]
        
        # Pad tokens and convert to tensors
        encoder_input_ids = [
            torch.tensor(context, dtype=torch.long)
            for context_list in encoder_input_ids
            for context in context_list
        ]
        encoder_input_ids = nn.utils.rnn.pad_sequence(
            encoder_input_ids, batch_first=True, padding_value=self.pad_token_id,
            padding_side="right"
        )
        encoder_attention_mask = encoder_input_ids.ne(self.pad_token_id).long()
        decoder_input_ids = [
            torch.tensor(context, dtype=torch.long)
            for context in decoder_input_ids
        ]
        decoder_input_ids = nn.utils.rnn.pad_sequence(
            decoder_input_ids, batch_first=True, padding_value=self.pad_token_id,
            padding_side="right"
        )
        decoder_attention_mask = decoder_input_ids.ne(self.pad_token_id).long()
        
        return (
            encoder_input_ids,
            encoder_attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            encoder_context_num,
            encoder_context_lengths
        )


class FidTrainingDataModule(L.LightningDataModule):
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 code_chunks_dir: str | None = None,
                 code_to_complete_dir: str | None = None,
                 code_chunks_filename_prefix: str = 'code-chunks_',
                 code_to_complete_filename_prefix: str = 'code-to-complete_',
                 tokenized_data_load_dir: str | None = None,
                 tokenized_data_save_dir: str | None = None,
                 encoder_max_tokens: int = 24576,
                 decoder_max_tokens: int = 8192,
                 code_chunk_formatter: BaseCodeChunkFormatter = PythonCommentCodeChunkFormatter(),
                 code_to_complete_formatter: BaseCodeToCompleteFormatter = PythonCommentCodeToCompleteFormatter(),
                 post_process_max_context_num: int | None = None,
                 copy_ratio: float = 0,

                 batch_size: int = 32,
                 ):
        super().__init__()
        self.dataset_kwargs = {
            'code_chunks_dir': code_chunks_dir,
            'code_to_complete_dir': code_to_complete_dir,
            'code_chunks_filename_prefix': code_chunks_filename_prefix,
            'code_to_complete_filename_prefix': code_to_complete_filename_prefix,
            'tokenizer': tokenizer,
            'tokenized_data_load_dir': tokenized_data_load_dir,
            'tokenized_data_save_dir': tokenized_data_save_dir,
            'encoder_max_tokens': encoder_max_tokens,
            'decoder_max_tokens': decoder_max_tokens,
            'code_chunk_formatter': code_chunk_formatter,
            'code_to_complete_formatter': code_to_complete_formatter,
            'post_process_max_context_num': post_process_max_context_num,
            'copy_ratio': copy_ratio,
        }
        self.batch_size = batch_size
        self.pad_token_id = tokenizer.pad_token_id

    
    def setup(self, stage: str):
        if stage == "fit":
            self.dataset = FidTrainingDataset(**self.dataset_kwargs)
            n_total = len(self.dataset)
            n_train = int(n_total * 0.95)
            n_val = n_total - n_train  # Ensure the sum matches exactly
            self.train_dataset, self.val_dataset = random_split(self.dataset, [n_train, n_val])
    

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=FidDataCollator(self.pad_token_id), num_workers=4,
                          drop_last=True)
    

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=FidDataCollator(self.pad_token_id), num_workers=4)




class RepeatFirstKSampler(Sampler):
    def __init__(self, k: int, total_length: int):
        self.k = k
        self.total_length = total_length

    def __iter__(self):
        return (i % self.k for i in range(self.total_length))

    def __len__(self):
        return self.total_length


class FidOverfitDataModule(FidTrainingDataModule):

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          sampler=RepeatFirstKSampler(16, 16000),
                          collate_fn=FidDataCollator(self.pad_token_id), num_workers=4,
                          drop_last=True)
