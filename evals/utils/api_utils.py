"""
This file defines various helper functions for interacting with the OpenAI API.
"""
import logging

import json
import backoff
import openai
import httpx


def generate_dummy_chat_completion():
    return {
        "id": "dummy-id",
        "object": "chat.completion",
        "created": 12345,
        "model": "dummy-chat",
        "usage": {"prompt_tokens": 56, "completion_tokens": 6, "total_tokens": 62},
        "choices": [
            {
                "message": {"role": "assistant", "content": "This is a dummy response."},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }


def generate_dummy_completion():
    return {
        "id": "dummy-id",
        "object": "text_completion",
        "created": 12345,
        "model": "dummy-completion",
        "choices": [
            {
                "text": "This is a dummy response.",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
    }


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.error.ServiceUnavailableError,
        openai.error.APIError,
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.Timeout,
    ),
    max_value=60,
    factor=1.5,
)
def openai_completion_create_retrying(*args, **kwargs):
    """
    Helper function for creating a completion.
    `args` and `kwargs` match what is accepted by `openai.Completion.create`.
    """
    if kwargs["model"] == "dummy-completion":
        return generate_dummy_completion()

    result = openai.Completion.create(*args, **kwargs)
    if "error" in result:
        logging.warning(result)
        raise openai.error.APIError(result["error"])
    return result


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.error.ServiceUnavailableError,
        openai.error.APIError,
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.Timeout,
    ),
    max_value=60,
    factor=1.5,
)
def openai_chat_completion_create_retrying(*args, **kwargs):
    """
    Helper function for creating a chat completion.
    `args` and `kwargs` match what is accepted by `openai.ChatCompletion.create`.
    """
    if kwargs["model"] == "dummy-chat":
        return generate_dummy_chat_completion()

    result = openai.ChatCompletion.create(*args, **kwargs)
    if "error" in result:
        logging.warning(result)
        raise openai.error.APIError(result["error"])
    return result


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        httpx.RequestError
    ),
    max_value=60,
    factor=1.5,
)
def xturing_completion_create_retrying(*args, **kwargs):
    """
    Helper function for creating a completion.
    `args` and `kwargs` match what is accepted by `openai.Completion.create`.
    """
    data = {
      "prompt": [kwargs["prompt"]],
      "params": {
        "penalty_alpha": 0.6,
        "top_k": 1.0,
        "top_p": 0.92,
        "do_sample": False,
        "max_new_tokens": 256,
        "temperature": 0.0,
      }
    }
    result = httpx.post(f'http://localhost:{kwargs["port"]}/api',
                        data=json.dumps(data),
                        headers={'Content-type': 'application/json'})
    result.raise_for_status()
    result = json.loads(result.text)
    if not result["success"]:
        logging.warning(result)
        raise httpx.RequestError

    res = result.pop("response")[0]
    result["choices"] = [res]
    return result


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        httpx.RequestError
    ),
    max_value=60,
    factor=1.5,
)
def huggingface_completion_create_retrying(*args, **kwargs):
    """
    Helper function for creating a completion.
    `args` and `kwargs` match what is accepted by `openai.Completion.create`.
    """
    data = {
      "prompt": kwargs["prompt"],
      "max_tokens": 256,
      "temperature": 1.0,
      "top_p": 1.0,
      "top_k": 50,
      "penalty_alpha": None,
      "repetition_penalty": 1.0,
    }
    result = httpx.post(f'http://localhost:{kwargs["port"]}/completions',
                        data=json.dumps(data),
                        headers={'Content-type': 'application/json'})
    result.raise_for_status()
    result = json.loads(result.text)
    return result


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        httpx.RequestError
    ),
    max_value=60,
    factor=1.5,
)
def huggingface_chat_completion_create_retrying(*args, **kwargs):
    """
    Helper function for creating a completion.
    `args` and `kwargs` match what is accepted by `openai.Completion.create`.
    """
    data = {
      "messages": kwargs["messages"],
      "max_tokens": 256,
      "temperature": 1.0,
      "top_p": 1.0,
      "top_k": 50,
      "penalty_alpha": None,
      "repetition_penalty": 1.0,
    }
    result = httpx.post(f'http://localhost:{kwargs["port"]}/chat/completions',
                        data=json.dumps(data),
                        headers={'Content-type': 'application/json'})
    result.raise_for_status()
    result = json.loads(result.text)
    return result
