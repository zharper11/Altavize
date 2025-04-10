import pandas as pd
import random
import math
import time
import json
import re
from collections import Counter
import logging
from typing import Dict, Any, Optional
import aiohttp
import sys
# Ensure the directory containing parallel_api.py is in the Python path
# Adjust the path as needed in your environment
from parallel_api import process_json  # The parallel processing function

# NEW OR MODIFIED CODE: Set your parallel API parameters
request_url = "https://api.openai.com/v1/chat/completions"
max_requests_per_minute = 250
max_tokens_per_minute = 30000
token_encoding_name = "cl100k_base"
max_attempts = 5
logging_level = 20

class CategoryAnalyzer:
    """
    A class to perform iterative categorization of items via GPT-like API calls,
    aggregating the responses and producing a final merged categorization.
    Uses parallel_api.process_json for parallel requests.
    """
    def __init__(self, openai_api_key: str, gpt_model: str = "gpt-4o") -> None:
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")
        self.api_key = openai_api_key
        self.gpt_model = gpt_model
        self.logger = logging.getLogger(__name__)
        
        # Track token usage
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        print("Initialized CategoryAnalyzer with model:", gpt_model)

    def _update_token_counts(self, usage: dict) -> None:
        """
        Update internal token counts based on 'usage' dict from the response
        where usage = {
            "prompt_tokens": <int>,
            "completion_tokens": <int>,
            "total_tokens": <int>
        }
        """
        try:
            self.input_tokens += usage["prompt_tokens"]
            self.output_tokens += usage["completion_tokens"]
            self.total_tokens += usage["total_tokens"]
        except KeyError:
            self.logger.warning("Token usage information is missing or incomplete in the response")

    # NEW OR MODIFIED CODE
    def build_parallel_request(self, prompt_text: str, row_id: int) -> dict:
        """
        Builds a single request JSON object for parallel_api.process_json
        """
        request_payload = {
            "model": self.gpt_model,
            "messages": [
                {"role": "user", "content": prompt_text}
            ],
            "max_tokens": 5000,
            "temperature": 0.7,
            "metadata": {"row_id": row_id},  # This helps us identify the response later
        }
        return request_payload

    def analyze_categories(
        self, df_item: pd.DataFrame, categorization_need: str = "Law Topics", category_number: int = 7
    ) -> Dict[str, Any]:
        """
        Analyze items in a DataFrame and return final aggregated category results along with token counts.
        Uses parallel processing for each iteration prompt.
        
        Expects the DataFrame to have an 'Item' column containing the items to be categorized.
        """
        print("Starting category analysis")
        
        # Calculate variant counts
        adjustment_number = round(category_number * (1 / 3))
        variants = {
            "lower": category_number - adjustment_number,
            "input": category_number,
            "upper": category_number + adjustment_number,
        }
        print("Variant counts set to:", variants)

        num_items = df_item.shape[0]
        if num_items < 100:
            sample_size = num_items
        else:
            sample_size = round(100 + ((num_items - 100) * 400) / 999900)
        print("Total items in DataFrame:", num_items, "Sample size per iteration:", sample_size)

        num_iterations = round((math.log(num_items) / math.log(sample_size)) * 5) if num_items > 1 else 1
        num_iterations = max(1, num_iterations)  # Ensure at least 1
        print("Number of iterations set to:", num_iterations)

        aggregated_counts = {variant: Counter() for variant in variants}

        # Build up a list of parallel requests for each iteration
        parallel_requests = []
        for run in range(1, num_iterations + 1):
            print(f"Iteration {run}/{num_iterations} - Sampling items")
            sample_df = df_item.sample(n=sample_size, random_state=random.randint(0, 100000))

            if "Item" not in sample_df.columns:
                self.logger.error("DataFrame must contain an 'Item' column")
                raise KeyError("DataFrame must contain an 'Item' column")

            items_list = sample_df["Item"].astype(str).tolist()

            # Build the prompt
            prompt_text = (
                f"Please analyze the following list of items and provide three sets of {categorization_need} that best summarize them. "
                "For each set, merge similar categories where appropriate. Provide one set with a lower number, one set with the standard number, "
                "and one set with an upper number of categories. The numbers are as follows:\n"
            )
            for variant, req_count in variants.items():
                prompt_text += f"- {variant.capitalize()}: {req_count}\n"
            prompt_text += "\nReturn your answer in valid JSON format with the following structure:\n\n"
            prompt_text += (
                "{\n"
                '  "lower": {\n'
                '    "requested_categories": <number for lower>,\n'
                '    "final_categories": [list of merged categories],\n'
                '    "output_class": "<insufficient/exact/excess>"\n'
                "  },\n"
                '  "input": {\n'
                '    "requested_categories": <number for standard>,\n'
                '    "final_categories": [list of merged categories],\n'
                '    "output_class": "<insufficient/exact/excess>"\n'
                "  },\n"
                '  "upper": {\n'
                '    "requested_categories": <number for upper>,\n'
                '    "final_categories": [list of merged categories],\n'
                '    "output_class": "<insufficient/exact/excess>"\n'
                "  }\n"
                "}\n\n"
                "Do not include any additional keys or commentary.\n\n"
                "Items:\n" + "\n".join(items_list)
            )

            # Add this request to the parallel queue
            # Use 'row_id' or 'request_id' to keep track of which iteration it belongs to
            request_obj = self.build_parallel_request(prompt_text, row_id=run)
            parallel_requests.append(request_obj)

        # Execute all iteration requests in parallel
        print("Sending iteration requests in parallel")
        iteration_responses = process_json(
            request_json=parallel_requests,
            request_url=request_url,
            api_key=self.api_key,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name=token_encoding_name,
            max_attempts=max_attempts,
            logging_level=logging_level,
        )

        # Parse iteration responses
        for response_data in iteration_responses:
            # response_data is typically a tuple of (original_request, api_response, metadata)
            if isinstance(response_data, Exception):
                self.logger.error(f"Exception occurred during parallel request: {response_data}")
                continue

            original_request, api_response, metadata = response_data
            row_id = metadata.get("row_id")  # This identifies which iteration this response came from

            # Update token usage
            usage = api_response.get("usage", {})
            self._update_token_counts(usage)

            # The GPT content
            try:
                content = api_response["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                self.logger.error(f"Malformed response in iteration {row_id}")
                continue

            # Clean and parse JSON
            clean_response = content.strip()
            clean_response = re.sub(r"^```(?:json)?\n", "", clean_response)
            clean_response = re.sub(r"\n```$", "", clean_response)

            try:
                result = json.loads(clean_response)
            except Exception:
                self.logger.warning(f"Failed to parse JSON response in iteration {row_id}")
                continue

            # Aggregate counts
            for variant in variants:
                variant_result = result.get(variant, {})
                final_categories = variant_result.get("final_categories", [])
                aggregated_counts[variant].update(final_categories)

        # Build aggregated text for final prompt
        aggregated_text_lower = "\n".join(f"{cat}: {count}" for cat, count in aggregated_counts["lower"].most_common())
        aggregated_text_input = "\n".join(f"{cat}: {count}" for cat, count in aggregated_counts["input"].most_common())
        aggregated_text_upper = "\n".join(f"{cat}: {count}" for cat, count in aggregated_counts["upper"].most_common())

        final_prompt = (
            f"Below are the aggregated distributions of {categorization_need} obtained from {num_iterations} runs for three variants:\n\n"
            f"Lower variant (requested categories = {variants['lower']}):\n{aggregated_text_lower}\n\n"
            f"Input variant (requested categories = {variants['input']}):\n{aggregated_text_input}\n\n"
            f"Upper variant (requested categories = {variants['upper']}):\n{aggregated_text_upper}\n\n"
            "Some categories might be very similar. Based on these aggregated distributions, please merge similar categories where appropriate "
            "and provide the best final categories for each variant. Return your answer in valid JSON format with the following structure:\n\n"
            "{\n"
            '  "lower": {\n'
            '    "requested_categories": <number for lower>,\n'
            '    "final_categories": [list of final merged categories],\n'
            '    "output_class": "<insufficient/exact/excess>"\n'
            "  },\n"
            '  "input": {\n'
            '    "requested_categories": <number for standard>,\n'
            '    "final_categories": [list of final merged categories],\n'
            '    "output_class": "<insufficient/exact/excess>"\n'
            "  },\n"
            '  "upper": {\n'
            '    "requested_categories": <number for upper>,\n'
            '    "final_categories": [list of final merged categories],\n'
            '    "output_class": "<insufficient/exact/excess>"\n'
            "  }\n"
            "}\n\n"
            "Do not include any additional keys or commentary."
        )

        # Send final aggregator prompt in parallel (just one request, but can still use process_json)
        final_request = [self.build_parallel_request(final_prompt, row_id=999999)]  # Arbitrary row_id
        final_response_data = process_json(
            request_json=final_request,
            request_url=request_url,
            api_key=self.api_key,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name=token_encoding_name,
            max_attempts=max_attempts,
            logging_level=logging_level,
            
        )

        final_result = None
        if final_response_data and not isinstance(final_response_data[0], Exception):
            _, final_api_response, final_metadata = final_response_data[0]
            self._update_token_counts(final_api_response.get("usage", {}))

            final_raw = final_api_response["choices"][0]["message"]["content"]
            final_clean = re.sub(r"^```(?:json)?\n", "", final_raw.strip())
            final_clean = re.sub(r"\n```$", "", final_clean)
            try:
                final_result = json.loads(final_clean)
            except Exception:
                self.logger.error("Failed to parse final JSON response")
                final_result = None

        print("Category analysis complete")
        return {
            "final_result": final_result,
            "token_counts": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
            },
        }

