import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import PreTrainedTokenizerBase
from .types import CodeChunk, CodeToComplete
from .formatter import BaseCodeChunkFormatter, BaseCodeToCompleteFormatter, PythonCommentCodeChunkFormatter, PythonCommentCodeToCompleteFormatter
import os
from tqdm import tqdm
import logging
import pickle
import lightning as L

logger = logging.getLogger(__name__)


class FidTrainingDataset(Dataset):
    """
    Dataset for training the FID model.
    This dataset yields:
    - encoder_input_ids: list[list[int]] - List of context chunks for the encoder. Each chunk is a list of token IDs.
    - decoder_input_ids: list[int] - List of token IDs for the decoder.
    """
    
    def __init__(self,
                 code_chunks_dir: str | None = None,
                 code_to_complete_dir: str | None = None,
                 code_chunks_filename_prefix: str = 'code-chunks_',
                 code_to_complete_filename_prefix: str = 'code-to-complete_',
                 tokenizer: PreTrainedTokenizerBase | None = None,
                 tokenized_data_load_dir: str | None = None,
                 tokenized_data_save_dir: str | None = None,
                 encoder_max_tokens: int = 24576,
                 decoder_max_tokens: int = 8192,
                 code_chunk_formatter: BaseCodeChunkFormatter = PythonCommentCodeChunkFormatter(),
                 code_to_complete_formatter: BaseCodeToCompleteFormatter = PythonCommentCodeToCompleteFormatter()):
        """
        You must provide either code_chunks_dir + code_to_complete_dir + tokenizer or tokenized_data_load_dir.
        If code_chunks_dir + code_to_complete_dir + tokenizer are provided, the dataset will tokenize the data and save it to tokenized_data_save_dir (if provided).
        If tokenized_data_load_dir is provided, the dataset will load the tokenized data from this directory.
        If both are provided, the dataset will use the tokenized data from tokenized_data_load_dir.
        """
        
        super().__init__()

        if (code_chunks_dir is None or code_to_complete_dir is None or tokenizer is None) and tokenized_data_load_dir is None:
            raise ValueError("You must provide either code_chunks_dir + code_to_complete_dir or tokenized_data_load_dir.")
        

        logger.info("Initializing FidTrainingDataset with the following parameters:")
        logger.info(f"code_chunks_dir: {code_chunks_dir}")
        logger.info(f"code_to_complete_dir: {code_to_complete_dir}")
        logger.info(f"code_chunks_filename_prefix: {code_chunks_filename_prefix}")
        logger.info(f"code_to_complete_filename_prefix: {code_to_complete_filename_prefix}")
        logger.info(f"tokenizer: {tokenizer.__class__.__name__}")
        logger.info(f"tokenized_data_load_dir: {tokenized_data_load_dir}")
        logger.info(f"tokenized_data_save_dir: {tokenized_data_save_dir}")
        logger.info(f"encoder_max_tokens: {encoder_max_tokens}")
        logger.info(f"decoder_max_tokens: {decoder_max_tokens}")
        logger.info(f"code_chunk_formatter: {code_chunk_formatter.__class__.__name__}")
        logger.info(f"code_to_complete_formatter: {code_to_complete_formatter.__class__.__name__}")
        
        if tokenized_data_load_dir is not None:
            if not os.path.exists(tokenized_data_load_dir):
                raise ValueError(f"tokenized_data_load_dir {tokenized_data_load_dir} does not exist.")
            self.encoder_input_ids, self.decoder_input_ids = self.load_tokenized_data(tokenized_data_load_dir)
        
        else:
            code_chunks, code_to_complete = self.load_raw_data(
                code_chunks_dir,
                code_to_complete_dir,
                code_chunks_filename_prefix,
                code_to_complete_filename_prefix
            )
            
            self.encoder_input_ids, self.decoder_input_ids = self.format_prompt_and_tokenize(
                code_chunks,
                code_to_complete,
                tokenizer,
                encoder_max_tokens,
                decoder_max_tokens,
                code_chunk_formatter,
                code_to_complete_formatter
            )

            if tokenized_data_save_dir is not None:
                self.save_tokenized_data(
                    self.encoder_input_ids,
                    self.decoder_input_ids,
                    tokenized_data_save_dir
                )


    def load_tokenized_data(self, tokenized_data_load_dir: str) -> tuple[list[list[int]], list[int]]:
        with open(os.path.join(tokenized_data_load_dir, 'tokenized_data.pkl'), 'rb') as f:
            data_dict = pickle.load(f)
        encoder_input_ids = data_dict['encoder_input_ids']
        decoder_input_ids = data_dict['decoder_input_ids']
        return encoder_input_ids, decoder_input_ids


    def save_tokenized_data(self, 
                            encoder_input_ids: list[list[int]],
                            decoder_input_ids: list[int],
                            tokenized_data_save_dir: str) -> None:
        data_dict = {
            'encoder_input_ids': encoder_input_ids,
            'decoder_input_ids': decoder_input_ids
        }
        if not os.path.exists(tokenized_data_save_dir):
            os.makedirs(tokenized_data_save_dir)
        with open(os.path.join(tokenized_data_save_dir, 'tokenized_data.pkl'), 'wb') as f:
            pickle.dump(data_dict, f)
        
    
    def load_raw_data(self, 
                  code_chunks_dir: str, 
                  code_to_complete_dir: str, 
                  code_chunks_filename_prefix: str, 
                  code_to_complete_filename_prefix: str) -> tuple[dict[int, CodeChunk], list[CodeToComplete]]:
        code_chunks = {}
        code_to_complete = []
        context_base_idx = 0  # Base index for the current repository.
        # This is used to ensure that the context IDs are unique across different repositories.

        code_to_complete_files = [f for f in os.listdir(code_to_complete_dir) if f.endswith('.jsonl')]
        if not code_to_complete_files:
            raise ValueError("No JSONL files found in the specified directories.")

        for filename in tqdm(code_to_complete_files, desc="FidTrainingDataset: Loading files"):
            if not filename.startswith(code_to_complete_filename_prefix):
                raise ValueError(f"Filename {filename} does not start with the expected prefix {code_to_complete_filename_prefix}.")
            code_chunks_filename = code_chunks_filename_prefix + filename[len(code_to_complete_filename_prefix):]

            if not os.path.exists(os.path.join(code_chunks_dir, code_chunks_filename)):
                raise ValueError(f"Code chunks file {code_chunks_filename} does not exist in the specified directory.")
            
            next_context_base_idx = context_base_idx

            with open(os.path.join(code_chunks_dir, code_chunks_filename), 'r') as f:
                for line in f:
                    code_chunk = CodeChunk.model_validate_json(line)

                    code_chunk.id += context_base_idx
                    next_context_base_idx = max(next_context_base_idx, code_chunk.id + 1)
                    code_chunks[code_chunk.id] = code_chunk

            with open(os.path.join(code_to_complete_dir, filename), 'r') as f:
                for line in f:
                    code = CodeToComplete.model_validate_json(line)
                    code.context = [c + context_base_idx for c in code.context]
                    code_to_complete.append(code)
        
            context_base_idx = next_context_base_idx

        return code_chunks, code_to_complete
    
    def format_prompt_and_tokenize(self,
            code_chunks: dict[int, CodeChunk],
            code_to_complete: list[CodeToComplete],
            tokenizer: PreTrainedTokenizerBase,
            encoder_max_tokens: int,
            decoder_max_tokens: int,
            code_chunk_formatter: BaseCodeChunkFormatter,
            code_to_complete_formatter: BaseCodeToCompleteFormatter) -> tuple[list[list[int]], list[int]]:
        encoder_input_ids_list = []
        decoder_input_ids_list = []

        for code in tqdm(code_to_complete, desc="FidTrainingDataset: Formatting and tokenizing"):
            # Encoder input IDs and attention mask
            num_encoder_tokens = 0
            encoder_input_ids = []
            for c in code.context:
                context_code_chunk = code_chunks[c]
                context_code_chunk_str = code_chunk_formatter(context_code_chunk)
                context_code_chunk_input_ids = tokenizer(context_code_chunk_str, truncation=True, max_length=encoder_max_tokens)['input_ids']
                num_encoder_tokens += len(context_code_chunk_input_ids)
                encoder_input_ids.append(context_code_chunk_input_ids)
                
                if num_encoder_tokens > encoder_max_tokens:
                    # truncate last context and break
                    encoder_input_ids[-1] = encoder_input_ids[-1][:encoder_max_tokens - (num_encoder_tokens - len(context_code_chunk_input_ids))]
                    break
            
            encoder_input_ids_list.append(encoder_input_ids)

            # Decoder input IDs and attention mask
            code_to_complete_str = code_to_complete_formatter(code)
            decoder_input_ids = tokenizer(code_to_complete_str, truncation=True, max_length=decoder_max_tokens)['input_ids']
            decoder_input_ids_list.append(decoder_input_ids)
        
        return encoder_input_ids_list, decoder_input_ids_list

    
    def __len__(self) -> int:
        return len(self.encoder_input_ids)
    
    def __getitem__(self, idx: int) -> tuple[list[list[int]], list[int]]:
        encoder_input_ids = self.encoder_input_ids[idx]
        decoder_input_ids = self.decoder_input_ids[idx]
        return encoder_input_ids, decoder_input_ids


class FidTrainingDataModule(L.LightningDataModule):
    def __init__(self,
                 code_chunks_dir: str | None = None,
                 code_to_complete_dir: str | None = None,
                 code_chunks_filename_prefix: str = 'code-chunks_',
                 code_to_complete_filename_prefix: str = 'code-to-complete_',
                 tokenizer: PreTrainedTokenizerBase | None = None,
                 tokenized_data_load_dir: str | None = None,
                 tokenized_data_save_dir: str | None = None,
                 encoder_max_tokens: int = 24576,
                 decoder_max_tokens: int = 8192,
                 code_chunk_formatter: BaseCodeChunkFormatter = PythonCommentCodeChunkFormatter(),
                 code_to_complete_formatter: BaseCodeToCompleteFormatter = PythonCommentCodeToCompleteFormatter(),
                 batch_size: int = 32):
        
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
            'code_to_complete_formatter': code_to_complete_formatter
        }
        self.batch_size = batch_size
    
    def setup(self, stage: str):
        if stage == "fit":
            self.dataset = FidTrainingDataset(**self.dataset_kwargs)
            self.train_dataset, self.val_dataset = random_split(self.dataset, [int(len(self.dataset) * 0.9), int(len(self.dataset) * 0.1)])
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)