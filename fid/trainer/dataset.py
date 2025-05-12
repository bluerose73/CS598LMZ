from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from ..data_util.types import CodeChunk, CodeToComplete
from ..data_util.formatter import BaseCodeChunkFormatter, BaseCodeToCompleteFormatter, PythonCommentCodeChunkFormatter, PythonCommentCodeToCompleteFormatter
import os
from tqdm import tqdm
import logging
import pickle
import random
from typing import Callable

logger = logging.getLogger(__name__)

def load_raw_data(
        code_chunks_dir: str, 
        code_to_complete_dir: str, 
        code_chunks_filename_prefix: str, 
        code_to_complete_filename_prefix: str,
        limit: int | None = None,
        code_chunk_parser: Callable[[str], CodeChunk] | None = None,
        code_to_complete_parser: Callable[[str], CodeToComplete] | None = None,
        ) -> tuple[dict[int, CodeChunk], list[CodeToComplete]]:
    code_chunks = {}
    code_to_complete = []
    context_base_idx = 0  # Base index for the current repository.
    # This is used to ensure that the context IDs are unique across different repositories.

    code_to_complete_files = [f for f in os.listdir(code_to_complete_dir) if f.endswith('.jsonl')]
    if not code_to_complete_files:
        raise ValueError("No JSONL files found in the specified directories.")
    if limit is not None:
        code_to_complete_files = code_to_complete_files[:limit]

    for filename in tqdm(code_to_complete_files, desc="FidTrainingDataset: Loading files"):
        if not filename.startswith(code_to_complete_filename_prefix):
            raise ValueError(f"Filename {filename} does not start with the expected prefix {code_to_complete_filename_prefix}.")
        code_chunks_filename = code_chunks_filename_prefix + filename[len(code_to_complete_filename_prefix):]

        if not os.path.exists(os.path.join(code_chunks_dir, code_chunks_filename)):
            raise ValueError(f"Code chunks file {code_chunks_filename} does not exist in the specified directory.")
        
        next_context_base_idx = context_base_idx

        with open(os.path.join(code_chunks_dir, code_chunks_filename), 'r') as f:
            for line in f:
                if code_chunk_parser is not None:
                    code_chunk = code_chunk_parser(line)
                else:
                    code_chunk = CodeChunk.model_validate_json(line)

                code_chunk.id += context_base_idx
                next_context_base_idx = max(next_context_base_idx, code_chunk.id + 1)
                code_chunks[code_chunk.id] = code_chunk

        with open(os.path.join(code_to_complete_dir, filename), 'r') as f:
            for line in f:
                if code_to_complete_parser is not None:
                    code = code_to_complete_parser(line)
                else:
                    code = CodeToComplete.model_validate_json(line)
                code.context = [c + context_base_idx for c in code.context]
                code_to_complete.append(code)
    
        context_base_idx = next_context_base_idx

    return code_chunks, code_to_complete

def format_prompt_and_tokenize(
        code_chunks: dict[int, CodeChunk],
        code_to_complete: list[CodeToComplete],
        tokenizer: PreTrainedTokenizerBase,
        encoder_max_tokens: int,
        decoder_max_tokens: int,
        max_context_num: int,
        max_context_length: int,
        code_chunk_formatter: BaseCodeChunkFormatter,
        code_to_complete_formatter: BaseCodeToCompleteFormatter) -> tuple[list[list[int]], list[int]]:
    encoder_input_ids_list = []
    decoder_input_ids_list = []

    for code in tqdm(code_to_complete, desc="FidTrainingDataset: Formatting and tokenizing"):
        # Encoder input IDs and attention mask
        num_encoder_tokens = 0
        encoder_input_ids = []

        for i, c in enumerate(code.context):
            if i >= max_context_num:
                break  # Limit the number of contexts to max_context_num

            context_code_chunk = code_chunks[c]
            context_code_chunk_str = code_chunk_formatter(context_code_chunk)

            context_code_chunk_input_ids = tokenizer(
                context_code_chunk_str, truncation=True, max_length=max_context_length
            )['input_ids']  # Limit individual context length to max_context_length

            num_encoder_tokens += len(context_code_chunk_input_ids)
            encoder_input_ids.append(context_code_chunk_input_ids)

            if num_encoder_tokens > encoder_max_tokens:
                # Truncate the last context to fit within encoder_max_tokens
                encoder_input_ids[-1] = encoder_input_ids[-1][:encoder_max_tokens - (num_encoder_tokens - len(context_code_chunk_input_ids))]
                break
        
        encoder_input_ids_list.append(encoder_input_ids)

        # Decoder input IDs and attention mask
        code_to_complete_str = code_to_complete_formatter(code)
        decoder_input_ids = tokenizer(
            code_to_complete_str, truncation=True, max_length=decoder_max_tokens
        )['input_ids']
        decoder_input_ids_list.append(decoder_input_ids)
    
    return encoder_input_ids_list, decoder_input_ids_list


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
                 encoder_max_tokens: int = 16384,
                 decoder_max_tokens: int = 2048,
                 max_context_num: int = 8,
                 max_context_length: int = 2048,
                 code_chunk_formatter: BaseCodeChunkFormatter = PythonCommentCodeChunkFormatter(),
                 code_to_complete_formatter: BaseCodeToCompleteFormatter = PythonCommentCodeToCompleteFormatter(),
                 post_process_max_context_num: int | None = None,
                 copy_ratio: float = 0,
                 limit: int | None = None,
                 code_chunk_parser: Callable[[str], CodeChunk] | None = None,
                 code_to_complete_parser: Callable[[str], CodeToComplete] | None = None,
                 ):
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
        logger.info(f"max_context_num: {max_context_num}")
        logger.info(f"max_context_length: {max_context_length}")
        logger.info(f"code_chunk_formatter: {code_chunk_formatter.__class__.__name__}")
        logger.info(f"code_to_complete_formatter: {code_to_complete_formatter.__class__.__name__}")
        
        if tokenized_data_load_dir is not None:
            if not os.path.exists(tokenized_data_load_dir):
                raise ValueError(f"tokenized_data_load_dir {tokenized_data_load_dir} does not exist.")
            self.encoder_input_ids, self.decoder_input_ids = self.load_tokenized_data(tokenized_data_load_dir)
        
        else:
            code_chunks, code_to_complete = load_raw_data(
                code_chunks_dir,
                code_to_complete_dir,
                code_chunks_filename_prefix,
                code_to_complete_filename_prefix,
                limit,
                code_chunk_parser=code_chunk_parser,
                code_to_complete_parser=code_to_complete_parser,
            )
            
            self.encoder_input_ids, self.decoder_input_ids = format_prompt_and_tokenize(
                code_chunks,
                code_to_complete,
                tokenizer,
                encoder_max_tokens,
                decoder_max_tokens,
                max_context_num,
                max_context_length,
                code_chunk_formatter,
                code_to_complete_formatter,
            )

            if tokenized_data_save_dir is not None:
                self.save_tokenized_data(
                    self.encoder_input_ids,
                    self.decoder_input_ids,
                    tokenized_data_save_dir
                )
        
        if post_process_max_context_num is not None:
            logger.info(f"Post-processing to limit the number of contexts to {post_process_max_context_num}")
            for i in tqdm(range(len(self.encoder_input_ids)), desc="FidTrainingDataset: Post-processing"):
                self.encoder_input_ids[i] = self.encoder_input_ids[i][:post_process_max_context_num]

        if copy_ratio > 0:
            for encoder_input_ids, decoder_input_ids in tqdm(zip(self.encoder_input_ids, self.decoder_input_ids),
                                                             desc="FidTrainingDataset: Copying decoder input ids to encoder input ids"):
                if random.random() < copy_ratio:
                    encoder_input_ids[-1] = decoder_input_ids


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
    
    def __len__(self) -> int:
        return len(self.encoder_input_ids)
    
    def __getitem__(self, idx: int) -> tuple[list[list[int]], list[int]]:
        encoder_input_ids = self.encoder_input_ids[idx]
        decoder_input_ids = self.decoder_input_ids[idx]
        return encoder_input_ids, decoder_input_ids



class FidTestDataset(Dataset):
    def __init__(self,
                 code_chunks_dir: str | None = None,
                 code_to_complete_dir: str | None = None,
                 code_chunks_filename_prefix: str = 'code-chunks_',
                 code_to_complete_filename_prefix: str = 'code-to-complete_',
                 tokenizer: PreTrainedTokenizerBase | None = None,
                 tokenized_data_load_dir: str | None = None,
                 tokenized_data_save_dir: str | None = None,
                 encoder_max_tokens: int = 16384,
                 decoder_max_tokens: int = 2048,
                 max_context_num: int = 8,
                 max_context_length: int = 2048,
                 code_chunk_formatter: BaseCodeChunkFormatter = PythonCommentCodeChunkFormatter(),
                 code_to_complete_formatter: BaseCodeToCompleteFormatter = PythonCommentCodeToCompleteFormatter(),
                 post_process_max_context_num: int | None = None,
                 code_chunk_parser: Callable[[str], CodeChunk] | None = None,
                 code_to_complete_parser: Callable[[str], CodeToComplete] | None = None,
                 ):
        """
        You must provide either code_chunks_dir + code_to_complete_dir + tokenizer or tokenized_data_load_dir.
        If code_chunks_dir + code_to_complete_dir + tokenizer are provided, the dataset will tokenize the data and save it to tokenized_data_save_dir (if provided).
        If tokenized_data_load_dir is provided, the dataset will load the tokenized data from this directory.
        If both are provided, the dataset will use the tokenized data from tokenized_data_load_dir.
        """
        
        super().__init__()

        if (code_chunks_dir is None or code_to_complete_dir is None or tokenizer is None) and tokenized_data_load_dir is None:
            raise ValueError("You must provide either code_chunks_dir + code_to_complete_dir or tokenized_data_load_dir.")
        

        logger.info("Initializing FidTestDataset with the following parameters:")
        logger.info(f"code_chunks_dir: {code_chunks_dir}")
        logger.info(f"code_to_complete_dir: {code_to_complete_dir}")
        logger.info(f"code_chunks_filename_prefix: {code_chunks_filename_prefix}")
        logger.info(f"code_to_complete_filename_prefix: {code_to_complete_filename_prefix}")
        logger.info(f"tokenizer: {tokenizer.__class__.__name__}")
        logger.info(f"tokenized_data_load_dir: {tokenized_data_load_dir}")
        logger.info(f"tokenized_data_save_dir: {tokenized_data_save_dir}")
        logger.info(f"encoder_max_tokens: {encoder_max_tokens}")
        logger.info(f"decoder_max_tokens: {decoder_max_tokens}")
        logger.info(f"max_context_num: {max_context_num}")
        logger.info(f"max_context_length: {max_context_length}")
        logger.info(f"code_chunk_formatter: {code_chunk_formatter.__class__.__name__}")
        logger.info(f"code_to_complete_formatter: {code_to_complete_formatter.__class__.__name__}")
        
        if tokenized_data_load_dir is not None:
            if not os.path.exists(tokenized_data_load_dir):
                raise ValueError(f"tokenized_data_load_dir {tokenized_data_load_dir} does not exist.")
            self.encoder_input_ids, self.decoder_input_ids, self.ground_truth_list = self.load_tokenized_data(tokenized_data_load_dir)
        
        else:
            code_chunks, code_to_complete = load_raw_data(
                code_chunks_dir,
                code_to_complete_dir,
                code_chunks_filename_prefix,
                code_to_complete_filename_prefix,
                code_chunk_parser=code_chunk_parser,
                code_to_complete_parser=code_to_complete_parser,
            )
            
            self.encoder_input_ids, self.decoder_input_ids = format_prompt_and_tokenize(
                code_chunks,
                code_to_complete,
                tokenizer,
                encoder_max_tokens,
                decoder_max_tokens,
                max_context_num,
                max_context_length,
                code_chunk_formatter,
                code_to_complete_formatter
            )

            self.ground_truth_list = [
                code_to_complete[i].metadata['ground_truth']
                for i in range(len(code_to_complete))
            ]

            if tokenized_data_save_dir is not None:
                self.save_tokenized_data(
                    self.encoder_input_ids,
                    self.decoder_input_ids,
                    self.ground_truth_list,
                    tokenized_data_save_dir
                )
        
        if post_process_max_context_num is not None:
            print(f"Post-processing to limit the number of contexts to {post_process_max_context_num}")
            for i in tqdm(range(len(self.encoder_input_ids)), desc="FidTestDataset: Post-processing"):
                self.encoder_input_ids[i] = self.encoder_input_ids[i][:post_process_max_context_num]
                


    def load_tokenized_data(self, tokenized_data_load_dir: str) -> tuple[list[list[int]], list[int], list[str]]:
        with open(os.path.join(tokenized_data_load_dir, 'tokenized_data.pkl'), 'rb') as f:
            data_dict = pickle.load(f)
        encoder_input_ids = data_dict['encoder_input_ids']
        decoder_input_ids = data_dict['decoder_input_ids']
        ground_truth_list = data_dict['ground_truth_list']
        return encoder_input_ids, decoder_input_ids, ground_truth_list


    def save_tokenized_data(self, 
                            encoder_input_ids: list[list[int]],
                            decoder_input_ids: list[int],
                            ground_truth_list: list[str],
                            tokenized_data_save_dir: str) -> None:
        data_dict = {
            'encoder_input_ids': encoder_input_ids,
            'decoder_input_ids': decoder_input_ids,
            'ground_truth_list': ground_truth_list
        }
        if not os.path.exists(tokenized_data_save_dir):
            os.makedirs(tokenized_data_save_dir)
        with open(os.path.join(tokenized_data_save_dir, 'tokenized_data.pkl'), 'wb') as f:
            pickle.dump(data_dict, f)
    
    def __len__(self) -> int:
        return len(self.encoder_input_ids)
    
    def __getitem__(self, idx: int) -> tuple[list[list[int]], list[int]]:
        encoder_input_ids = self.encoder_input_ids[idx]
        decoder_input_ids = self.decoder_input_ids[idx]
        ground_truth = self.ground_truth_list[idx]
        return encoder_input_ids, decoder_input_ids, ground_truth

