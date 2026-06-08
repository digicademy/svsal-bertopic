import httpx
import time
import argparse


def wait_for_server_to_come_up(
    url: str,
    timeout: int = 300,
    time_between_retries: int = 3,
) -> int:
    """Sleep until the server responds and return its status code.
    Raises httpx.ConnectError if the timeout is reached.

    Args:
        url: The server URL to poll (e.g. http://localhost:11434).
        timeout: Maximum seconds to wait.
        time_between_retries: Seconds between connection attempts.

    Returns:
        HTTP status code of the first successful response.
    """
    print(f"Waiting for the server at {url} to come up ...")
    start_time = time.time()
    while True:
        try:
            response = httpx.get(url)
            if response.is_success:
                print(f"Server at {url} is up. Status code: {response.status_code}")
            else:
                print(
                    f"Server at {url} responded but might have issues. "
                    f"Status code: {response.status_code}"
                )
            return response.status_code
        except httpx.ConnectError as error:
            if time.time() - start_time > timeout:
                print(
                    f"Timeout reached. The server at {url} is still not up "
                    f"after {timeout}s."
                )
                raise error
            time.sleep(time_between_retries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        help="The URL to poll (Ollama default: http://localhost:11434).",
        default="http://localhost:11434",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Maximum seconds to wait before giving up (default: 300).",
    )
    args = parser.parse_args()
    wait_for_server_to_come_up(args.url, timeout=args.timeout)
