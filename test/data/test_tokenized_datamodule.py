from fid.trainer.fid_datamodule import FidTrainingDataModule, FidTrainingDataset
import lightning as L
from transformers.models.qwen2 import Qwen2TokenizerFast

L.seed_everything(42)

tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")

datamodule = FidTrainingDataModule(
    tokenized_data_load_dir="/work/nvme/becw/sma2/the-stack-v2-20k/tokenized",
    tokenizer=tokenizer,
    batch_size=8,
    copy_ratio=1,
    post_process_max_context_num=1,
)

datamodule.setup("fit")


dataset = datamodule.train_dataset

# Print the first 5 examples of the dataset
for i in range(5):
    print(f"============================= dataset[{i}] ===============================")
    encoder_input_ids_list, decoder_input_ids = dataset[i]
    print("---------------------- encoder_input_list -------------------------")
    for j, encoder_input_ids in enumerate(encoder_input_ids_list):
        print(f"encoder_input_ids[{j}]:")
        print(tokenizer.decode(encoder_input_ids))
    print("---------------------- decoder_input ------------------------------")
    print(tokenizer.decode(decoder_input_ids))
    print()