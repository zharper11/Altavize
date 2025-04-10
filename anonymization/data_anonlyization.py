from openai import OpenAI
import pandas as pd
import json
from pathlib import Path
import os
import logging
from typing import Dict, Any, Optional, List
import time

class DataAnonymizer:
    def __init__(self, openai_api_key: str):
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")
            
        self.client = OpenAI(api_key=openai_api_key)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        # Use the provided absolute path for prompts directory
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_file_path))
        self.prompts_dir = Path(os.path.join(project_root, "analysis", "Prompt"))
        
        if not self.prompts_dir.exists():
            raise ValueError(f"Prompts directory not found at {self.prompts_dir}")
        
        # Initialize prompt file mapping with correct filenames
        self.prompt_files = {
            "name": "Name.txt",
            "email": "Email_address.txt",
            "address": "Location.txt",
            "job_title": "Job_title.txt",
            "date": "Data_of_birth.txt",
            "employer": "Employer.txt",
            "salary": "Salary.txt",
           
        }
        
        # Validate prompt files exist
        self._validate_prompt_files()
        
        # Track processed values for consistency within a record
        self.processed_values = {}
    def _update_token_counts(self, usage):
        """
        Update internal token counts based on usage from OpenAI response
        """
        if not usage:
            return
            
        # Handle CompletionUsage object - it has attributes instead of dictionary keys
        if hasattr(usage, 'prompt_tokens'):
            self.input_tokens += usage.prompt_tokens
            self.output_tokens += usage.completion_tokens
            self.total_tokens += usage.total_tokens
        else:
            # Fallback for dictionary-like objects
            self.input_tokens += usage.get('prompt_tokens', 0)
            self.output_tokens += usage.get('completion_tokens', 0)
            self.total_tokens += usage.get('total_tokens', 0)
        

    def _validate_prompt_files(self) -> None:
        """Validate that all prompt files exist"""
        missing_files = []
        #logging.info("Validating prompt files...",{self.prompts_dir.absolute()})
        self.logger.info(f"Checking prompt files in directory: {self.prompts_dir}")
        for field_type, filename in self.prompt_files.items():
            file_path = self.prompts_dir / filename
            if not file_path.exists():
                missing_files.append(filename)
                self.logger.warning(f"Missing prompt file: {file_path}")
        
        if missing_files:
            self.logger.warning(f"Missing prompt files: {', '.join(missing_files)}")

    def load_prompt(self, prompt_type: str) -> Optional[str]:
        """Load prompt from file with improved error handling"""
        try:
            file_path = self.prompts_dir / self.prompt_files.get(prompt_type, "")
            if not file_path.exists():
                self.logger.warning(f"Prompt file not found: {file_path}")
                return None
                
            with open(file_path, 'r', encoding='utf-8') as file:
                prompt = file.read().strip()
                
            return prompt
                
        except Exception as e:
            self.logger.error(f"Error loading prompt {prompt_type}: {str(e)}")
            return None

    def get_gpt_response(self, prompt: str, max_retries: int = 3) -> str:
        """Get response from GPT model with retry logic"""
        retry_count = 0
      
        while retry_count < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a data anonymization assistant. "
                                                     "For address data, return properly structured JSON objects without markdown formatting. "
                                                     "Do not use ```json tags. "
                                                      "For SSN values, ALWAYS anonymize ALL numeric sequences that appear in the input: "
                                                      "- Treat ANY sequence of digits as a potential SSN requiring anonymization "
                                                      "- This includes 9-digit numbers, but also any other numeric sequence regardless of length "
                                                      "- Replace each digit with a randomly generated digit (0-9) "
                                                      "- Preserve the exact format including any separators (hyphens, spaces) "
                                                      "- Always generate different random numbers than the original "
                                                      "- When 'SSN' is mentioned explicitly before numbers, ensure ALL following numbers are anonymized "
                                                      "- Ensure consistent replacement if the same number appears multiple times "
                                                     "Salary data should remain the same not anonymized."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5
                )
                result = response.choices[0].message.content.strip()
                # Track token usage
                if hasattr(response, 'usage'):
                    self._update_token_counts(response.usage)
                return result
                
            except Exception as e:
                retry_count += 1
                wait_time = 2 ** retry_count  # Exponential backoff
                self.logger.warning(f"GPT API error (attempt {retry_count}/{max_retries}): {str(e)}. Retrying in {wait_time}s")
                if retry_count == max_retries:
                    self.logger.error(f"GPT API error after {max_retries} retries: {str(e)}")
                    raise
                time.sleep(wait_time)

    def create_anonymization_prompt(self, value: Any, field_type: str, field_name: str) -> str:
        """Create a customized prompt for anonymization based on field type"""
         # Initial logging of input parameters
        self.logger.info(f"Creating anonymization prompt for field '{field_name}' of type '{field_type}' with value: {value}")
        prompt_template = self.load_prompt(field_type)
        if not prompt_template:
            self.logger.warning(f"No prompt template found for {field_type}. Using fallback approach.")
            return f"Anonymize this {field_type}: {value}. Return only the anonymized value."
        
        # Convert value to string and sanitize
        value_str = str(value).strip()
        
        # Replace placeholders in the prompt template based on field type
        prompt = prompt_template
        
        # Common replacements
        if "[Value]" in prompt:
            prompt = prompt.replace("[Value]", value_str)
        if "[Field]" in prompt:
            prompt = prompt.replace("[Field]", field_name)
        if "[Initial Header]" in prompt:
            prompt = prompt.replace("[Initial Header]", field_name)
            
        # Field-specific replacements
        if field_type == "name" and "[Name]" in prompt:
            prompt = prompt.replace("[Name]", value_str)
        elif field_type == "email" and "[Email Address]" in prompt:
            prompt = prompt.replace("[Email Address]", value_str)
            # Add dependency for name anonymization if needed
            if "name" in self.processed_values and "[Dependency]" in prompt:
                prompt = prompt.replace("[Dependency]", self.processed_values.get("name", "Anonymous User"))
        elif field_type == "date" and "[Date of Birth]" in prompt:
            prompt = prompt.replace("[Date of Birth]", value_str)
        


        elif field_type == "job_title" and "[Job Title]" in prompt:
            prompt = prompt.replace("[Job Title]", value_str)  # Replace job title placeholder
            #logging.info(f"Job Title awais: {value_str},{prompt}") 
        elif "[Employer]" in prompt:
            prompt = prompt.replace("[Employer]", value_str)

      



        
        elif field_type == "address" and "[Location Data]" in prompt:
            prompt = prompt.replace("[Location Data]", value_str)
            prompt = prompt.replace("[Dimensions]", str(len(value_str.split(','))))
        elif field_type == "location" and "[Location_csv]" in prompt:
            prompt = prompt.replace("[Location_csv]", value_str)
        elif field_type == "salary":
            if "[Salary]" in prompt:
                prompt = prompt.replace("[Salary]", value_str)
            else:
                prompt += f"\nAnonymize this salary: {value_str}\nProvide a realistic but anonymized salary value."
        
        # Add a clear instruction for all prompts
        if "Return only the anonymized value" not in prompt:
            prompt += "\nReturn only the anonymized value without explanation or formatting."
            
        return prompt

    def _clean_gpt_response(self, response: str, field_type: str) -> Any:
        """Clean up GPT response to extract only the anonymized value"""
        # Remove any explanations or extra text
        response = response.strip()
        
        # Handle responses wrapped in markdown code blocks
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        elif response.startswith("```") and response.endswith("```"):
            response = response[3:-3].strip()
        
        # For JSON responses (common with address anonymization)
        if response.startswith('{') and response.endswith('}'):
            try:
                json_data = json.loads(response)
                if field_type == "address" and isinstance(json_data, dict):
                    return json_data
                elif field_type == "address":
                    return {"Address": str(json_data)}
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse JSON response for {field_type}: {response}")
                # Fall through to regular string processing
        
        # For non-JSON responses, proceed with standard cleaning
        if ":" in response and not (field_type == "address" and len(response) > 100):
            # Take the part after the last colon
            response = response.split(":")[-1].strip()
            
        # Remove quotes
        response = response.strip('"\'')
        
        return response

    def anonymize_field(self, value: Any, field_type: str, field_name: str) -> Any:
        """Anonymize a single field using GPT"""
        self.logger.info(f"BEFORE PROMPT CREATION - Field: {field_name}, Type: {field_type}, Value: {value}")

        if not value or (isinstance(value, str) and not value.strip()):
            return ""
            
        if not isinstance(value, (str, int, float, bool)):
            self.logger.warning(f"Converting {field_name} of type {type(value)} to string")
            value = str(value)
        
        try:
            # Create appropriate prompt from the text file
        
            prompt = self.create_anonymization_prompt(value, field_type, field_name)
            
            self.logger.info(f"AFTER PROMPT CREATION - Field: {field_name}, Prompt length: {len(prompt)}")
            self.logger.debug(f"Generated prompt for {field_name} (truncated): {prompt[:100]}...")
            
            anonymized_value = self.get_gpt_response(prompt)
            
            # Clean up the response
            cleaned_value = self._clean_gpt_response(anonymized_value, field_type)
            
            # Store in processed values for consistency within the current record
            if field_type == "name":
                self.processed_values["name"] = cleaned_value
                
            return cleaned_value
            
        except Exception as e:
            self.logger.error(f"Error anonymizing {field_type} '{value}': {str(e)}")
            
            

    def anonymize_data(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Main anonymization function with input validation"""
        if not isinstance(input_data, dict):
            raise ValueError("Input data must be a dictionary")
            
        data = input_data.get("data", [])
        config = input_data.get("columnsConfig", {})
        
        if not isinstance(data, list):
            raise ValueError("Data must be a list of records")
            
        anonymized_data = []
        total_records = len(data)
        
        for i, record in enumerate(data):
            if i % 10 == 0:
                self.logger.info(f"Anonymizing record {i+1}/{total_records}")
                
            if not isinstance(record, dict):
                raise ValueError("Each record must be a dictionary")
                
            # Reset processed values for each record to maintain consistency within a record
            # but allow variation between records
            self.processed_values = {}
            
            anonymized_record = {}
            for field_name, value in record.items():
                field_config = config.get(field_name, {})
                field_type = field_config.get("type", "none")
                self.logger.info(f"FIELD CONFIG - Name outside if: '{field_name}', Type: '{field_type}'")

                self.logger.info(f"ANONYMIZING FIELD - Name inside if: '{field_name}', Type: '{field_type}'")
                anonymized_record[field_name] = self.anonymize_field(
                    value, field_type, field_name
                )

                    
            anonymized_data.append(anonymized_record)
        
        return anonymized_data

    def anonymize_dataset(self, dataset_df: Any, categories_df: Optional[Any] = None) -> Any:
        """Anonymize a dataset with improved type handling"""
        try:
            # Convert DataFrame to the format expected by anonymize_data
            if isinstance(dataset_df, pd.DataFrame):
                data = dataset_df.to_dict('records')
            else:
                data = dataset_df

            # Create columnsConfig from categories_df
            token_metrics = {
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.total_tokens
                                    }
            self.logger.info(f"Anonymization complete. Token usage: {token_metrics}")

            config = {}
            if categories_df is not None:
                if isinstance(categories_df, pd.DataFrame):
                    for _, row in categories_df.iterrows():
                        col_name = row.get("columnName")
                        anon_type = row.get("anonymizationType", "none")
                        if col_name:
                            config[col_name] = {"type": anon_type}
                else:
                    # Assume it's a dictionary-like structure
                    for col, details in categories_df.items():
                        if isinstance(details, dict):
                            config[col] = {"type": details.get("type", "none")}
                        else:
                            config[col] = {"type": "none"}

            input_data = {
                "data": data,
                "columnsConfig": config
            }

            self.logger.info(f"Starting anonymization of {len(data)} records with {len(config)} field configurations")
            result = self.anonymize_data(input_data)
            self.logger.info(f"Anonymization completed for {len(result)} records")
            
            # Convert back to DataFrame if input was DataFrame
            if isinstance(dataset_df, pd.DataFrame):
                return pd.DataFrame(result)
            return {
                "anonymized_data": result,
                "token_counts": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                    "total_tokens": self.total_tokens
                }
            }

        except Exception as e:
            self.logger.error(f"Dataset anonymization error: {str(e)}")
            raise

