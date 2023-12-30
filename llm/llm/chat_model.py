import asyncio
import json
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import config
import httpx
from llm_utils import ProgressBar
from openai import AsyncAzureOpenAI, AzureOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from typing_extensions import Self

chat_logger = logging.getLogger(name="ChatLogger")


@dataclass(kw_only=True)
class Message:
    "Wrapper around messages that packages metadata with message."
    messages: list[dict[str, str]]
    metadata: dict[str, Any]


@dataclass(kw_only=True)
class Bundle:
    "Input & output messages bundled with metadata & API call settings."
    # ID Created by API Call
    id: str | None = None
    # Input Messages
    system_message: str | None = None
    user_message: str | None = None
    # Metadata packaged with Message
    metadata: dict | None = None
    # Response Message
    response_message: str | None = None
    # Response Metadata
    created_time: int | None = None
    model: str | None = None
    total_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    response_format: dict | None = None
    finish_reason: str | None = None
    system_fingerprint: str | None = None
    # Entire API Response object as a dict
    response_dict: dict | None = None
    # Additional API Call Arguments
    seed: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    # Track Num Retries
    num_retries: int | None = None
    # Callback Response
    _callback_response: Any | None = None

    @property
    def callback_response(self) -> Any | None:
        return self._callback_response

    @callback_response.setter
    def callback_response(self, callback_response: Any) -> Self:
        self._callback_response = callback_response


# Type Aliases
MessagesType = list[dict[str, str]] | Message
ChatCompletionResponseType = ChatCompletion | str | Bundle | dict[str, Any] | None


# Example Function Signature for callback functions
def validation_callback(
    messages: MessagesType,
    response: ChatCompletionResponseType,
) -> bool:
    """Override/substitute this with user-defined validation of the response.
    Return `True` to accept response and `False` to reject response."""
    return True


def callback(
    messages: MessagesType,
    response: ChatCompletionResponseType,
) -> Any | None:
    """Override/substitute this with user-defined logic."""
    pass


@dataclass(kw_only=True)
class ChatModel:
    """Wrapper around OpenAI ChatCompletions API with retry, output validation,
    output_formatting, sync and async implementations, and concurrency limits on async.
    """

    # OpenAI API Config
    sync_client: AzureOpenAI = AzureOpenAI(
        api_key=config.USWEST_OPENAI_API_KEY,
        azure_endpoint=config.USWEST_OPENAI_API_ENDPOINT,
        api_version=config.USWEST_OPENAI_API_VERSION,
        max_retries=10,
        timeout=httpx.Timeout(180.0),
    )
    async_client: AsyncAzureOpenAI = AsyncAzureOpenAI(
        api_key=config.USWEST_OPENAI_API_KEY,
        azure_endpoint=config.USWEST_OPENAI_API_ENDPOINT,
        api_version=config.USWEST_OPENAI_API_VERSION,
        max_retries=10,
        timeout=httpx.Timeout(180.0),
    )
    # Model Config
    model: str = "gpt-35-turbo-1106"  # "gpt-4-1106"
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = None
    n: int = 1
    seed: int | None = 42
    response_format: dict[str, str] | None = field(default_factory=lambda: {"type": "json_object"})

    def create_chat_completion(
        self, system_message: str, user_message: str, **kwargs
    ) -> ChatCompletionResponseType:
        """Simplified Chat Completion call that packages `system_message` and `user_message`
        for us.

        Args:
            system_message (str): system message prompt
            user_message (str): user message prompt
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        parsed_cc = self.chat_completion(messages=messages, **kwargs)
        return parsed_cc

    def parse_chat_completion_response(
        self,
        cc: ChatCompletion,
        output_format: str | None = "bundle_dict",
        messages: MessagesType | None = None,
        **kwargs,
    ) -> ChatCompletion | str | Bundle | dict:
        """Parse ChatCompletion object.

        Args:
            output_format (str | None, optional): Controls format of output.
                `raw` or `None`: return raw ChatCompletion object with no modification.
                `simple`: return only response message
                `bundle`: return namedtuple with input+output messages, message metadata,
                    ChatCompletion metadata, and other `kwargs` flattened as a namedtuple
                `bundle_dict`: same as `bundle`, but returns as a dictionary.
            messages (MessagesType | None, optional): The original input messages.
                If this is a Message object, the metadata will be unpacked and
                parsed in the `bundle` and `bundle_dict` output.

        Returns:
            Either ChatCompletion, string response message, Bundle, or dict depending
            on `output_format`.
        """
        # If given `messages`, split out system_message and user_message to separate fields
        if messages is not None:
            if isinstance(messages, Message):  # unpack Message object
                msgs = messages.messages
                metadata = messages.metadata
            else:  # `messages` is the raw list[dict[str, str]] messages
                msgs = messages
                metadata = None
            system_message = [m for m in msgs if m["role"] == "system"][0]
            user_message = [m for m in msgs if m["role"] == "user"][0]
            kwargs |= {
                "system_message": system_message["content"],
                "user_message": user_message["content"],
                "metadata": metadata,
            }

        # Remove from kwargs to avoid duplicate if they are present in ChatCompletion
        for key in ("model", "n"):
            if key in kwargs:
                kwargs.pop(key)

        match output_format:
            case "simple":
                if cc is None:
                    return None
                else:
                    chat_completion_message = cc.choices[0].message.content
                    return chat_completion_message
            case "bundle" | "bundle_dict":
                if cc is None:
                    if output_format == "bundle_dict":
                        return Bundle(**kwargs)._asdict()
                    else:
                        return Bundle(**kwargs)
                else:
                    bundle = Bundle(
                        id=cc.id,
                        response_message=cc.choices[0].message.content,
                        created_time=cc.created,
                        model=cc.model,
                        total_tokens=cc.usage.total_tokens,
                        prompt_tokens=cc.usage.prompt_tokens,
                        completion_tokens=cc.usage.completion_tokens,
                        finish_reason=cc.choices[0].finish_reason,
                        system_fingerprint=cc.system_fingerprint,
                        response_dict=json.loads(cc.model_dump_json()),
                        **kwargs,
                    )
                    if output_format == "bundle_dict":
                        return bundle._asdict()
                    else:
                        return bundle
            case "raw" | None:
                return cc
            case _:
                return cc

    def chat_completion(
        self,
        messages: MessagesType,
        output_format: str | None = None,
        num_retries: int = 3,
        validation_callback: Callable[
            [MessagesType, ChatCompletionResponseType], bool
        ] = validation_callback,
        callback: Callable[[MessagesType, ChatCompletionResponseType], Any] | None = None,
        **kwargs,
    ) -> ChatCompletionResponseType:
        """Calls OpenAI ChatCompletions API.
        https://platform.openai.com/docs/api-reference/chat/create

        This method uses properties declared on class as default arguments.
        Any keyword arguments directly passed in to `kwargs` will override
        the default arguments.

        Args:
            messages (MessagesType): List of dict message format
                or a `Message` wrapper for the original `messages` can be accessed
                at `Message.messages`.
                ```python
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of the USA?"},
                ],
                ```
            output_format (str | None, optional): Controls format of output.
                see method `parse_chat_completion_response`.
                NOTE: If message metadata or callback response is desired in output,
                then output_format=`bundle` or `bundle_dict` must be selected.
            num_retries (int): Number of retries if API call fails.  If still fails,
                then `None` is returned.
            validation_callback (Callable | None, optional): A function that accepts
                the input `messages`, and `response` (the result of formatting the raw
                ChatCompletion to the selected `output_format`) and returns `True` or `False`.
                If `True`, will accept ChatCompletion response and proceed.
                If `False`, ChatCompletion response is rejected and the and will proceed
                to retry if `num_retries` > 0.
                This callback should be used to add any logic to check whether or not
                a ChatCompletion response is acceptable prior to returning the result.
            callback (Callable | None, optional): A user-defined callback function
                that accepts the input `messages`, and `response` which will be called
                upon passing `validation_callback`.

        Returns:
            Either ChatCompletion, string response message, Bundle, or dict depending
            on `output_format`.  If API call fails, returns `None`.
        """
        default_kwargs = {
            "messages": messages,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "n": self.n,
            "seed": self.seed,
            "response_format": self.response_format,
        }
        updated_kwargs = default_kwargs | kwargs

        def attempt_retry(
            messages: MessagesType, response: ChatCompletionResponseType, num_retries: int
        ) -> ChatCompletionResponseType:
            """Attempts retry for chat completion if num_retries > 0.
            Otherwise, if num_retries = 0, will invoke callback and format output response.
            If no callback is provided, then `invoke callback` is a no-op and the
            formatted ChatCompletion response bundle is returned."""
            kwargs = updated_kwargs | {"seed": self.seed + 1}
            if num_retries > 0:
                # Decrement retry counter, recursively call this method
                return self.chat_completion(
                    **kwargs,
                    output_format=output_format,
                    num_retries=num_retries,
                    validation_callback=validation_callback,
                    callback=callback,
                )
            else:
                return invoke_callback(messages, response)

        def invoke_callback(
            messages: MessagesType, response: ChatCompletionResponseType
        ) -> ChatCompletionResponseType:
            "Call User-Defined Callback if any, add to bundle, return response"
            if callback is not None and isinstance(callback, Callable):
                callback_response = callback(messages, response)
            if isinstance(response, Bundle):
                response.callback_response = callback_response
            elif isinstance(response, dict):
                response["callback_response"] = callback_response
            return response

        response = None
        try:
            # Format kwargs for API call
            api_kwargs = updated_kwargs.copy()
            msgs = api_kwargs.pop("messages")
            if isinstance(msgs, Message):
                msgs = msgs.messages
            cc = self.sync_client.chat.completions.create(messages=msgs, **api_kwargs)
            # Format API call response
            response = self.parse_chat_completion_response(
                cc=cc,
                output_format=output_format,
                messages=messages,
                num_retries=num_retries,
                **api_kwargs,
            )
            # Validation Callback
            did_pass_validation = validation_callback(messages, response)
            if did_pass_validation:
                return invoke_callback(messages, response)
            else:
                return attempt_retry(messages, response, num_retries=num_retries - 1)
        except Exception as e:
            warnings.warn(
                f"Failed to create ChatCompletion with arguments: {updated_kwargs.items()}\n"
                f"Exception: {e}\n"
                f"Retries left: {num_retries}"
            )
            return attempt_retry(messages, response, num_retries=num_retries - 1)

    def chat_completions(
        self,
        messages_list: list[MessagesType],
        description: str = "ChatCompletions",
        **kwargs,
    ) -> list[ChatCompletionResponseType]:
        "Calls `chat_completion` multiple times and returns a list of ChatCompletion objects."
        cc_list = []
        with ProgressBar() as p:
            for message in p.track(messages_list, description=description):
                cc = self.chat_completion(messages=message, **kwargs)
                cc_list += [cc]
        return cc_list

    async def async_chat_completion(
        self,
        messages: MessagesType,
        output_format: str | None = None,
        num_retries: int = 3,
        validation_callback: Callable[
            [MessagesType, ChatCompletionResponseType], bool
        ] = validation_callback,
        callback: Callable[[MessagesType, ChatCompletionResponseType], Any] | None = None,
        **kwargs,
    ) -> ChatCompletionResponseType:
        "Same as `chat_completion` but using asynchronous (non-blocking) client."
        default_kwargs = {
            "messages": messages,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "n": self.n,
            "seed": self.seed,
            "response_format": self.response_format,
        }
        updated_kwargs = default_kwargs | kwargs

        async def attempt_retry(
            messages: MessagesType, response: ChatCompletionResponseType, num_retries: int
        ) -> ChatCompletionResponseType:
            """Attempts retry for chat completion if num_retries > 0.
            Otherwise, if num_retries = 0, will invoke callback and format output response.
            If no callback is provided, then `invoke callback` is a no-op and the
            formatted ChatCompletion response bundle is returned."""
            kwargs = updated_kwargs | {"seed": self.seed + 1}
            if num_retries > 0:
                # Decrement retry counter, recursively call this method
                return await self.async_chat_completion(
                    **kwargs,
                    output_format=output_format,
                    num_retries=num_retries,
                    validation_callback=validation_callback,
                    callback=callback,
                )
            else:
                return await invoke_callback(messages, response)

        async def invoke_callback(
            messages: MessagesType, response: ChatCompletionResponseType
        ) -> ChatCompletionResponseType:
            "Call User-Defined Callback if any, add to bundle, return response"
            if callback is not None and isinstance(callback, Callable):
                callback_response = callback(messages, response)
            if isinstance(response, Bundle):
                response.callback_response = callback_response
            elif isinstance(response, dict):
                response["callback_response"] = callback_response
            return response

        response = None
        try:
            # Format kwargs for API call
            api_kwargs = updated_kwargs.copy()
            msgs = api_kwargs.pop("messages")
            if isinstance(msgs, Message):
                msgs = msgs.messages
            cc = await self.async_client.chat.completions.create(messages=msgs, **api_kwargs)

            # Format API call response
            response = self.parse_chat_completion_response(
                cc=cc,
                output_format=output_format,
                messages=messages,
                num_retries=num_retries,
                **api_kwargs,
            )
            # Validation Callback
            did_pass_validation = validation_callback(messages, response)
            if did_pass_validation:
                return await invoke_callback(messages, response)
            else:
                return await attempt_retry(messages, response, num_retries=num_retries - 1)
        except Exception as e:
            warnings.warn(
                f"Failed to create ChatCompletion with arguments: {updated_kwargs.items()}\n"
                f"Exception: {e}\n"
                f"Retries left: {num_retries}"
            )
            return await attempt_retry(messages, response, num_retries=num_retries - 1)

    async def async_chat_completions(
        self,
        messages_list: list[MessagesType],
        num_concurrent: int = 5,
        timeout: int | None = None,
        description: str = "ChatCompletions",
        **kwargs,
    ) -> list[ChatCompletionResponseType]:
        """Calls `async_chat_completion` multiple times and returns a list of
        ChatCompletion objects. Concurrency is controlled using `num_concurrent`."""

        async def generation_task(semaphore, messages, **kwargs) -> ChatCompletionResponseType:
            "Wrap ChatCompletion API call with a blocking semaphore to control concurrency."
            async with semaphore:
                cc = await self.async_chat_completion(messages=messages, **kwargs)
                return cc

        async def generate_concurrent() -> list[ChatCompletionResponseType]:
            "Main task to schedule on asyncio event loop."
            # Create the shared semaphore
            semaphore = asyncio.BoundedSemaphore(num_concurrent)
            # Create and schedule tasks, limiting concurrent tasks with semaphore
            task_list = []
            for messages in messages_list:
                task = asyncio.create_task(
                    generation_task(semaphore=semaphore, messages=messages, **kwargs)
                )
                task_list += [task]
            # Await each task to complete with progress bar (returns in order of completion)
            with ProgressBar() as p:
                for task in p.track(
                    asyncio.as_completed(task_list),
                    description=description,
                    total=len(task_list),
                ):
                    await task
            # Await to ensure all tasks are done
            await asyncio.wait(task_list)
            # Return results in original order of tasks
            cc_list = [await task for task in task_list]
            return cc_list

        # Start the asyncio program
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # 'RuntimeError: There is no current event loop...'
            loop = None

        # Schedule coroutine as a task on event loop if it already exists,
        # otherwise run coroutine on a new event loop
        if loop and loop.is_running():
            tsk = loop.create_task(generate_concurrent())
            await asyncio.wait_for(tsk, timeout=timeout)
            result = tsk.result()
        else:
            result = asyncio.run(generate_concurrent())
        return result
