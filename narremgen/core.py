"""
narremgen.core
=============
Core utility functions: OpenAI API calls, token estimation, and robust output saving.
"""

import os, time
import pandas as pd
from io import StringIO
from pathlib import Path
from getpass import getpass
from openai import OpenAI, OpenAIError

def get_openai_key(verbose=False):
    """
    Retrieve and persist the OpenAI API key for authenticated access.

    This utility searches for an existing API key in the following order:
    1. Environment variable `OPENAI_API_KEY`.
    2. A local file `~/.narremgen_key.txt`.
    3. Interactive user input (masked).  
    If obtained interactively, the key can optionally be saved for future sessions.
    The function validates that the key format begins with `sk-` before storing it.

    Parameters
    ----------
    verbose : bool, optional
        If True, prints where the key was loaded from or saved to. Default: False.

    Returns
    -------
    str
        The validated OpenAI API key string.

    Raises
    ------
    ValueError
        If an entered key does not appear to be a valid OpenAI API key
        (i.e., it does not start with 'sk-').

    Notes
    -----
    - The retrieved key is written into the process environment to ensure
    subsequent API calls succeed without reloading credentials.
    - This helper is safe to call multiple times; it will reuse the cached key if found.
    """

    key = os.getenv("OPENAI_API_KEY")
    if key and key.startswith("sk-"):
        return key

    keyfile = Path.home() / ".narremgen_key.txt"
    if keyfile.exists():
        key = keyfile.read_text(encoding="utf-8").strip()
        if key.startswith("sk-"):
            os.environ["OPENAI_API_KEY"] = key
            if verbose:
                print(f"Key loaded from {keyfile}")
            return key

    if verbose:
        print("No OpenAI Key found!!")
        print("Please, enter the Key now with the keyboard (it will kept confidential)!!")
    key = getpass("Write your OpenAI Key (begin with 'sk-') : ").strip()
    if not key.startswith("sk-"):
        raise ValueError("!! Not valid OpenAI Key.")

    save = input("Do you want to store the key in the file ~/.narremgen_key.txt for later use? [o/N] ").strip().lower()
    if save == "o":
        keyfile.write_text(key, encoding="utf-8")
        if verbose:
            print(f"Key stored in {keyfile}")

    os.environ["OPENAI_API_KEY"] = key
    return key


def safe_chat_completion(
    client,
    model: str,
    messages: list[dict],
    max_tokens: int = 4000,
    temperature: float | None = None,
    verbose = False,
    **kwargs
):
    """
    Send a robust chat completion request with backward compatibility and error fallback.

    This helper ensures compatibility across OpenAI API versions by first attempting a call
    using the `max_completion_tokens` parameter and falling back to the older `max_tokens`
    if needed. It also handles transient API errors by retrying with reduced verbosity.

    Parameters
    ----------
    client : openai.OpenAI
        Authenticated OpenAI client instance.
    model : str
        Model name to use (e.g., 'gpt-4o-mini', 'gpt-4o', 'o1', 'o3').
    messages : list[dict]
        List of chat message dictionaries, e.g. `[{"role": "system", "content": "..."}, ...]`.
    max_tokens : int, optional
        Token limit for the completion (default 4000).
    temperature : float | None, optional
        Sampling temperature (ignored for deterministic or reasoning models). Default: None.
    verbose : bool, optional
        If True, prints detailed information about the API call and any fallback attempts.
    **kwargs :
        Additional arguments forwarded to the OpenAI API (e.g., `tools`, `response_format`).

    Returns
    -------
    str | None
        The content of the model’s message on success, or None if all attempts failed.

    Notes
    -----
    - This wrapper prevents most compatibility crashes when switching between newer and older model families.
    - The function returns `None` rather than raising an exception to simplify pipeline fault tolerance.
    """

    def call_with(param_name: str, verbose = False):
        try:
            if verbose:
                print(f"safe_chat_completion: model={model} using {param_name}={max_tokens}")
            token_kwargs = {param_name: max_tokens}
            token_kwargs.update(kwargs)

            temp_for_api = None
            if ("gpt-4o" in model) or ("o1" in model):
                temp_for_api = 1
            else:
                if temperature is not None:
                    temp_for_api = temperature

            call_kwargs = dict(model=model, messages=messages, **token_kwargs)
            if temp_for_api is not None:
                call_kwargs["temperature"] = temp_for_api

            resp = client.chat.completions.create(**call_kwargs)
            return resp.choices[0].message.content

        except OpenAIError as e:
            print(f"!! safe_chat_completion OpenAIError with {param_name}: {e}")
            raise
        except Exception as e:
            print(f"!! safe_chat_completion unexpected error with {param_name}: {e}")
            raise

    try:
        return call_with("max_completion_tokens", verbose=verbose)
    except OpenAIError as e:
        txt = str(e).lower()
        if "unsupported parameter" in txt and "max_completion_tokens" in txt:
            print("API rejected 'max_completion_tokens' - retrying with 'max_tokens'.")
            try:
                return call_with("max_tokens", verbose=verbose)
            except Exception as e2:
                print(f"!!! safe_chat_completion both attempts failed: {e2}")
                return None
        else:
            print("!!! OpenAIError (non param-mismatch) - aborting safe_chat_completion.")
            return None
    except Exception as e:
        print(f"!!! safe_chat_completion fatal unexpected error: {e}")
        return None


def generate_text(prompt: str, client: OpenAI, model_name="gpt-4o-mini", 
                  temperature: float | None = None, max_tokens=8000,
                  verbose=False) -> str:
    """
    Execute a single chat-based text generation request with automatic fallback parameters.

    This function builds on `safe_chat_completion` to send one well-formed prompt to a model,
    managing both modern (`max_completion_tokens`) and legacy (`max_tokens`) parameter sets.
    It returns the generated text string or None if both attempts fail. The purpose is to
    provide a unified and predictable interface for low-level text generation within the
    Narremgen pipeline.

    Parameters
    ----------
    prompt : str
        The textual prompt to be sent to the model.
    client : openai.OpenAI
        Authenticated OpenAI client instance.
    model_name : str, optional
        Model name to use for the request (default 'gpt-4o-mini').
    temperature : float | None, optional
        Optional temperature for creative variance. Ignored by deterministic models. Default: None.
    max_tokens : int, optional
        Maximum number of completion tokens to request (default 8000).
    verbose : bool, optional
        If True, prints internal retry details and any raised API errors.

    Returns
    -------
    str | None
        The generated text returned by the model, or None if the call failed after all retries.

    Notes
    -----
    - This low-level helper is used throughout the package wherever direct prompt-to-text
    generation is needed (e.g., Advice, Mapping, Context, or Narrative stages).
    - It is tolerant to temporary API failures and fallback errors, ensuring that higher-level
    functions can gracefully continue without terminating the pipeline.
    """

    def try_call(param_name: str, param_value: int, verbose=False):
        try:
            kwargs = {param_name: param_value}
            if verbose: print(f"generate_text: calling model={model_name} with {param_name}={param_value}")

            temp_for_api = None
            if ("gpt-4o" in model_name) or ("o1" in model_name):
                temp_for_api = 1
            else:
                if temperature is not None:
                    temp_for_api = temperature

            call_kwargs = dict(model=model_name,
                               messages=[{"role": "user", "content": prompt}],
                               **kwargs)
            if temp_for_api is not None:
                call_kwargs["temperature"] = temp_for_api

            response = client.chat.completions.create(**call_kwargs)
            if response and getattr(response, "choices", None):
                return response.choices[0].message.content
            return None

        except OpenAIError as e:
            if verbose: print(f"!! generate_text OpenAIError with {param_name}: {e}")
            raise
        except Exception as e:
            if verbose: print(f"!! generate_text unexpected error with {param_name}: {e}")
            raise

    try:
        return try_call("max_completion_tokens", max_tokens)
    except OpenAIError as e:
        txt = str(e).lower()
        if "unsupported parameter" in txt and "max_completion_tokens" in txt:
            print("API rejected 'max_completion_tokens' - retrying with 'max_tokens'.")
            try:
                return try_call("max_tokens", max_tokens)
            except Exception as e2:
                if verbose: print(f"!!! Both attempts failed: {e2}")
                return None
        else:
            if verbose: print("!!! OpenAIError (non param-mismatch) - aborting generate_text.")
            return None
    except Exception:
        return None


def estimate_tokens(giga_prompt: str, batch_size: int, target_words: int, 
                    model_name: str, max_tokens: int | None = None):
    """
    Estimate the number of tokens required for a narrative generation prompt.

    This function provides a rough heuristic for token budgeting by combining
    prompt length, target batch size, and expected words per narrative. It is
    used to warn when a batch prompt might exceed a model’s context window.

    Parameters
    ----------
    giga_prompt : str
        The complete prompt text that will be sent to the model.
    batch_size : int
        The number of individual narratives to include in the batch.
    target_words : int
        Expected number of words per generated narrative.
    model_name : str
        Model name (used to infer default context limits for that model family).
    max_tokens : int | None, optional
        Explicit upper token cap for the model. If None, defaults are inferred
        from the `model_name` family (e.g., 128k for GPT-4o).

    Returns
    -------
    tuple[int, int, int, int, int]
        A tuple containing:
        (total_estimate, prompt_tokens, expected_output_tokens, max_model_tokens, max_tokens)
        where:
        - `total_estimate`: approximate total tokens used,
        - `prompt_tokens`: estimated token count for the input prompt,
        - `expected_output_tokens`: expected tokens to generate,
        - `max_model_tokens`: the maximum context length for the model family,
        - `max_tokens`: the requested generation limit.

    Notes
    -----
    - This estimation is heuristic and should be treated as a safeguard, not a guarantee.
    - It helps maintain API stability and prevent truncation in large batch runs.
    """

    avg_chars_per_token = 4
    prompt_tokens = len(giga_prompt) // avg_chars_per_token
    
    # 1 token ≈ 0.75 words -> inverse ratio to estimate required tokens
    expected_output_tokens = int(batch_size * (target_words / 0.75))
    
    max_model_tokens = 8000 if "mini" in model_name else 128000
    if max_tokens is None:
        max_tokens = max_model_tokens

    total_estimate = prompt_tokens + expected_output_tokens
    return total_estimate, prompt_tokens, expected_output_tokens, max_model_tokens, max_tokens
