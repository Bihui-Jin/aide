"""Backend for OpenAI API."""

from asyncio import subprocess
import json
import logging
import os
import socket
import struct
import time

from aide.backend.utils import (
    FunctionSpec,
    OutputType,
    opt_messages_to_list,
    backoff_create,
)
from funcy import notnone, once, select_values
import openai

logger = logging.getLogger("aide")

_client: openai.OpenAI = None  # type: ignore


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def get_docker_host_ip() -> str:
    """
    Resolve the Docker host gateway IP (i.e. the host machine's IP as seen from
    inside a Linux bridge-networked container).

    Resolution order:
      1. DOCKER_HOST_IP env var (injected by run_agent.py or manually)
      2. /proc/net/route default gateway (most reliable on Linux containers)
      3. host.docker.internal (Docker Desktop / macOS fallback)
      4. 172.17.0.1 (hard fallback for standard Linux bridge)
    """
    # # 1. Explicit override wins
    # explicit = os.environ.get("DOCKER_HOST_IP", "").strip()
    # if explicit:
    #     logger.info(f"Docker host IP from env DOCKER_HOST_IP: {explicit}")
    #     return explicit

    # 2. Read default gateway from /proc/net/route (Linux only)
    try:
        with open("/proc/net/route") as f:
            for line in f.readlines()[1:]:          # skip header
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                destination = parts[1]
                gateway_hex = parts[2]
                flags = int(parts[3], 16)
                # 0x0002 = RTF_GATEWAY, 0x0001 = RTF_UP
                # destination == "00000000" means the default route
                if destination == "00000000" and (flags & 0x0003) == 0x0003:
                    # /proc/net/route stores IPs in little-endian hex
                    gateway_ip = socket.inet_ntoa(
                        struct.pack("<I", int(gateway_hex, 16))
                    )
                    logger.info(f"Docker host IP from /proc/net/route gateway: {gateway_ip}")
                    return gateway_ip
    except Exception as e:
        logger.warning(f"Could not read /proc/net/route: {e}")

    # 3. Docker Desktop / macOS
    try:
        resolved = socket.gethostbyname("host.docker.internal")
        logger.info(f"Docker host IP from host.docker.internal: {resolved}")
        return resolved
    except Exception:
        pass

    # 4. Standard Linux bridge hard fallback
    logger.warning("Falling back to hard-coded Docker bridge IP: 172.17.0.1")
    return "172.17.0.1"

@once
def _setup_openai_client():
    global _client
    docker_host_ip = get_docker_host_ip()
    a = subprocess.run(["curl", f"http://{docker_host_ip}:8000/v1/models"], capture_output=True, text=True)
    logger.info(f"curl result: {a.stdout}")
    logger.info(f"Resolved Docker host IP as: {docker_host_ip}")
    _client = openai.OpenAI(max_retries=0, base_url=f'http://{docker_host_ip}:8000/v1', api_key="testkey")



def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_openai_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_message, convert_system_to_user=convert_system_to_user)

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model the use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    t0 = time.time()
    completion = backoff_create(
        _client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        output = choice.message.content
    else:
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            output = json.loads(choice.message.tool_calls[0].function.arguments)
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
