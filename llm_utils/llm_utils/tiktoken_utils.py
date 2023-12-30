import logging

import tiktoken

# OpenAI Utility Functions
# Modified From: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

log = logging.getLogger("UtilityLogger")
log.setLevel(logging.WARNING)


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def num_tokens_from_messages(messages, model="gpt-3.5-turbo") -> int:
    """Return the number of tokens used by a list of messages.
    This is a modified version of function provided by OpenAI:
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    try:
        if model in {
            # Custom model names, which use clk100k_base tokenizer encoding
            "gpt4",
            "gpt4_8k",
            "gpt4_32k",
            "gpt35turbo-0301",
            "gpt35turbo",
        }:
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        # OpenAI defined model names
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        # Custom model names, which map to GPT-4
        "gpt4",
        "gpt4_8k",
        "gpt4_32k",
        "gpt-35-turbo-1106",
        "gpt-4-1106",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model in {
        # OpenAI defined model names
        "gpt-3.5-turbo-0301",
        # Custom model names, which map to GPT-3.5-turbo
        "gpt35turbo-0301",
        "gpt35turbo",
    }:
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. "
            "Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. "
            "Returning num tokens assuming gpt-4-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"num_tokens_from_messages() is not implemented for model {model}. "
            "See https://github.com/openai/openai-python/blob/main/chatml.md for "
            "information on how messages are converted to tokens."
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def truncate_string_tail(string: str, encoding_name: str, max_token_length: int) -> str:
    """Truncates string to max token length by removing tokens from end of string."""
    encoding = tiktoken.get_encoding(encoding_name)
    encoded_string = encoding.encode(string)
    truncated_encoded_string = encoded_string[:max_token_length]
    truncated_string = encoding.decode(truncated_encoded_string)
    return truncated_string


GPT35TURBO_4K_MAX_TOKENS = 4096
GPT35TURBO_16K_MAX_TOKENS = 16384
GPT4_8K_MAX_TOKENS = 8192
GPT4_32K_MAX_TOKENS = 32768


def get_model_max_token_length(model_name: str) -> int:
    match model_name:
        case "gpt35turbo_4k" | "gpt35turbo":
            max_token_length = GPT35TURBO_4K_MAX_TOKENS
        case "gpt35turbo_16k":
            max_token_length = GPT35TURBO_16K_MAX_TOKENS
        case "gpt4_8k" | "gpt4":
            max_token_length = GPT4_8K_MAX_TOKENS
        case "gpt4_32k":
            max_token_length = GPT4_32K_MAX_TOKENS
        case _:
            raise ValueError("Unknown `model_name` {model_name}.")
    return max_token_length


def get_system_message_length(
    system_message: str, encoding_name: str = "cl100k_base"
) -> int:
    "Token length of system message + additional tokens in message format."
    system_message_token_length = num_tokens_from_string(
        system_message, encoding_name=encoding_name
    )
    # Additional Formatting Tokens
    tokens_per_message = 4
    reply_prefix_tokens = 3
    # Get Total Tokens Used by System Message & Message Format
    system_message_used_tokens = (
        system_message_token_length + tokens_per_message * 2 + reply_prefix_tokens
    )
    return system_message_used_tokens
