def process_json(
    request_json,
    request_url,
    api_key,
    max_requests_per_minute,
    max_tokens_per_minute,
    token_encoding_name,
    max_attempts,
    logging_level
):
    # imports
    import aiohttp  # for making API calls concurrently
    import argparse  # for running script from command line
    import asyncio  # for running API calls concurrently
    import json  # for saving results to a jsonl file
    import logging  # for logging rate limit warnings and other messages
    import os  # for reading API key
    import re  # for matching endpoint from request URL
    import tiktoken  # for counting tokens
    import time  # for sleeping after rate limit is hit
    from dataclasses import dataclass, field  # for storing API inputs, outputs, and metadata

    # dataclasses

    @dataclass
    class StatusTracker:
        """Stores metadata about the script's progress. Only one instance is created."""
        num_tasks_started: int = 0
        num_tasks_in_progress: int = 0  # script ends when this reaches 0
        num_tasks_succeeded: int = 0
        num_tasks_failed: int = 0
        num_rate_limit_errors: int = 0
        num_api_errors: int = 0  # excluding rate limit errors, counted above
        num_other_errors: int = 0
        time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits

    @dataclass
    class APIRequest:
        """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""
        task_id: int
        request_json: dict
        token_consumption: int
        attempts_left: int
        metadata: dict
        result: list = field(default_factory=list)

        async def call_api(
            self,
            session: aiohttp.ClientSession,
            request_url: str,
            request_header: dict,
            retry_queue: asyncio.Queue,
            status_tracker: StatusTracker,
        ):
            """Calls the OpenAI API and saves results."""
            logging.info(f"Starting request #{self.task_id}")
            error = None
            try:
                async with session.post(
                    url=request_url, headers=request_header, json=self.request_json
                ) as response:
                    response = await response.json()
                if "error" in response:
                    logging.warning(
                        f"Request {self.task_id} failed with error {response['error']}"
                    )
                    status_tracker.num_api_errors += 1
                    error = response
                    if "rate limit" in response["error"].get("message", "").lower():
                        status_tracker.time_of_last_rate_limit_error = time.time()
                        status_tracker.num_rate_limit_errors += 1
                        status_tracker.num_api_errors -= 1  # track rate-limit errors separately
            except Exception as e:
                logging.warning(f"Request {self.task_id} failed with Exception {e}")
                status_tracker.num_other_errors += 1
                error = e

            if error:
                self.result.append(error)
                if self.attempts_left:
                    retry_queue.put_nowait(self)
                else:
                    logging.error(
                        f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}"
                    )
                    data = (
                        [self.request_json, [str(e) for e in self.result], self.metadata]
                        if self.metadata
                        else [self.request_json, [str(e) for e in self.result]]
                    )
                    append_to_json_object(data, json_object)
                    status_tracker.num_tasks_in_progress -= 1
                    status_tracker.num_tasks_failed += 1
            else:
                data = (
                    [self.request_json, response, self.metadata]
                    if self.metadata
                    else [self.request_json, response]
                )
                append_to_json_object(data, json_object)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_succeeded += 1

    async def process_api_requests_from_file(
        request_json,
        request_url,
        api_key,
        max_requests_per_minute,
        max_tokens_per_minute,
        token_encoding_name,
        max_attempts,
        logging_level,
    ):
        """Processes API requests in parallel, throttling to stay under rate limits."""
        # constants
        seconds_to_pause_after_rate_limit_error = 15
        seconds_to_sleep_each_loop = 0.001  # 1 ms limits throughput

        # initialize logging
        logging.basicConfig(level=logging_level)
        logging.debug(f"Logging initialized at level {logging_level}")

        # infer API endpoint and construct request header
        api_endpoint = api_endpoint_from_url(request_url)
        request_header = {"Authorization": f"Bearer {api_key}"}
        if "/deployments" in request_url:
            request_header = {"api-key": f"{api_key}"}

        # initialize trackers
        queue_of_requests_to_retry = asyncio.Queue()
        task_id_generator = task_id_generator_function()  # note: second file used a different variable name here
        status_tracker = StatusTracker()
        next_request = None

        # initialize available capacity counts and time tracking for updates
        available_request_capacity = max_requests_per_minute
        available_token_capacity = max_tokens_per_minute
        last_update_time = time.time()

        # initialize file reading using an iterator (renamed for clarity as in the second file)
        requests_iter = iter(request_json)
        logging.debug("Initialization complete. Entering main loop")

        async with aiohttp.ClientSession() as session:
            while True:
                # get next request (if one is not already waiting for capacity)
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                        logging.debug(
                            f"Retrying request {next_request.task_id}: {next_request}"
                        )
                    else:
                        try:
                            single_request_json = next(requests_iter)
                            next_request = APIRequest(
                                task_id=next(task_id_generator),
                                request_json=single_request_json,
                                token_consumption=num_tokens_consumed_from_request(
                                    single_request_json, api_endpoint, token_encoding_name
                                ),
                                attempts_left=max_attempts,
                                metadata=single_request_json.pop("metadata", None),
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logging.debug(
                                f"Reading request {next_request.task_id}: {next_request}"
                            )
                        except StopIteration:
                            logging.debug("All requests have been read")

                # update available capacity based on elapsed time
                current_time = time.time()
                seconds_since_update = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity
                    + max_requests_per_minute * seconds_since_update / 60.0,
                    max_requests_per_minute,
                )
                available_token_capacity = min(
                    available_token_capacity
                    + max_tokens_per_minute * seconds_since_update / 60.0,
                    max_tokens_per_minute,
                )
                last_update_time = current_time

                # if enough capacity is available, dispatch the next request
                if next_request:
                    next_request_tokens = next_request.token_consumption
                    if available_request_capacity >= 1 and available_token_capacity >= next_request_tokens:
                        available_request_capacity -= 1
                        available_token_capacity -= next_request_tokens
                        next_request.attempts_left -= 1

                        asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue_of_requests_to_retry,
                                status_tracker=status_tracker,
                            )
                        )
                        next_request = None

                # if all tasks are finished, break
                if status_tracker.num_tasks_in_progress == 0:
                    break

                # sleep briefly so concurrent tasks can run
                await asyncio.sleep(seconds_to_sleep_each_loop)

                # if a rate limit error was hit recently, pause to cool down
                seconds_since_rate_limit_error = time.time() - status_tracker.time_of_last_rate_limit_error
                if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
                    remaining_seconds_to_pause = seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error
                    await asyncio.sleep(remaining_seconds_to_pause)
                    logging.warn(
                        f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
                    )

        # after finishing, log final status
        logging.info("Parallel processing complete. Results sent to main application")
        if status_tracker.num_tasks_failed > 0:
            logging.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to main application"
            )
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate"
            )
        return json_object

    def api_endpoint_from_url(request_url):
        """Extract the API endpoint from the request URL."""
        match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
        if match is None:
            # for Azure OpenAI deployment urls
            match = re.search(r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\\?|$)", request_url)
        return match[1]

    def append_to_json_object(data, json_object: list) -> None:
        """Append a JSON payload to a local JSON-like object."""
        json_object.append(data)

    def num_tokens_consumed_from_request(request_json: dict, api_endpoint: str, token_encoding_name: str):
        """Count the number of tokens in the request. Only supports completion and embedding requests."""
        encoding = tiktoken.get_encoding(token_encoding_name)
        # if completions request, tokens = prompt + n * max_tokens
        if api_endpoint.endswith("completions"):
            max_tokens = request_json.get("max_tokens", 15)
            n = request_json.get("n", 1)
            completion_tokens = n * max_tokens

            # chat completions
            if api_endpoint.startswith("chat/"):
                num_tokens = 0
                for message in request_json["messages"]:
                    num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                    for key, value in message.items():
                        if key == "content":
                            # Handle content which might be a list for multimodal inputs
                            if isinstance(value, list):
                                for item in value:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        # Only encode text items
                                        text_content = item.get("text", "")
                                        if isinstance(text_content, str):
                                            num_tokens += len(encoding.encode(text_content))
                                    # For images, add a fixed token count estimate
                                    elif isinstance(item, dict) and item.get("type") == "image_url":
                                        # Estimate: images cost approximately 1000 tokens each
                                        # This is a conservative estimate and may need adjustment
                                        num_tokens += 1000
                            elif isinstance(value, str):
                                num_tokens += len(encoding.encode(value))
                        elif isinstance(value, str):
                            num_tokens += len(encoding.encode(value))
                            if key == "name":  # if there's a name, the role is omitted
                                num_tokens -= 1  # role is always required and always 1 token
                        # Skip token counting for non-string values that aren't handled above
                num_tokens += 2  # every reply is primed with <im_start>assistant
                return num_tokens + completion_tokens
            # normal completions
            else:
                prompt = request_json["prompt"]
                if isinstance(prompt, str):  # single prompt
                    prompt_tokens = len(encoding.encode(prompt))
                    return prompt_tokens + completion_tokens
                elif isinstance(prompt, list):  # multiple prompts
                    prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                    return prompt_tokens + completion_tokens * len(prompt)
                else:
                    raise TypeError('Expecting either string or list of strings for "prompt" field in completion request')
        # if embeddings request, tokens = input tokens
        elif api_endpoint == "embeddings":
            input_data = request_json["input"]
            if isinstance(input_data, str):  # single input
                return len(encoding.encode(input_data))
            elif isinstance(input_data, list):  # multiple inputs
                return sum([len(encoding.encode(i)) for i in input_data])
            else:
                raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')
        else:
            # If API endpoint is not recognized, return a safe default value
            # This prevents crashes while allowing the code to continue
            return 4000  # Conservative estimate
    def task_id_generator_function():
        """Generate integers 0, 1, 2, and so on."""
        task_id = 0
        while True:
            yield task_id
            task_id += 1

    # Return JSON file with results
    json_object = []
    json_object = asyncio.run(
        process_api_requests_from_file(
            request_json=request_json,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name=token_encoding_name,
            max_attempts=max_attempts,
            logging_level=logging_level
        )
    )

    return json_object