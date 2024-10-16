from torchtune.data import Message
from typing import Any, Callable, Dict, List, Mapping, Optional
from torchtune.datasets import ChatDataset
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.models.llama3 import llama3_tokenizer
from functools import partial

def message_converter(sample: Mapping[str, Any], train_on_input: bool) -> List[Message]:
    input_msg = sample["input"]
    output_msg = sample["output"]

    user_message = Message(
        role="user",
        content=input_msg,
        masked=not train_on_input,  # Mask if not training on prompt
    )
    assistant_message = Message(
        role="assistant",
        content=output_msg,
        masked=False,
    )
    # A single turn conversation
    messages = [user_message, assistant_message]

    return messages

def custom_dataset(
    *,
    tokenizer: ModelTokenizer,
    max_seq_len: int = 8192,
) -> ChatDataset:

    return ChatDataset(
        tokenizer=tokenizer,
        source="json",
        split="train",
        convert_to_messages=message_converter,
        chat_format=None,
        max_seq_len=max_seq_len,
        data_files="", # Replace with your training data directory
    )

dataset = partial(custom_dataset)
dataset.__doc__ = """custom dataset for RoBBR"""