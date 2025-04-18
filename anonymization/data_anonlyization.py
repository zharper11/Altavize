from openai import OpenAI
import pandas as pd
import json
from pathlib import Path
import os
import logging
from typing import Dict, Any, Optional, List
import time
from countryinfo import CountryInfo
import spacy
import pycountry
from countryinfo import CountryInfo
from langdetect import detect, LangDetectException
from faker import Faker
import re

# Cache for spaCy model to avoid reloading
_nlp = None

def get_nlp():
    """Get or initialize the spaCy model with caching"""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

# List of problematic locales that should be avoided
PROBLEMATIC_LOCALES = [
    'ar_AA', 'ar_AE', 'ar_BH', 'ar_EG', 'ar_JO', 'ar_PS',  # Arabic locales often cause issues
    'zh_CN', 'zh_TW',  # Chinese locales may have formatting issues
    'ja_JP',  # Japanese locales may have formatting issues
    'ko_KR',  # Korean locales may have formatting issues
    'th', 'th_TH',  # Thai locales may have formatting issues
]

def normalize_country_name(country_name: str) -> str:
    """
    Normalize country names to improve matching with pycountry and CountryInfo.
    Handles common variations and abbreviations.
    """
    if not country_name:
        return ""
    
    # Convert to lowercase for comparison
    country_name = country_name.lower().strip()
    
    # Common country name mappings
    country_mappings = {
        'usa': 'united states',
        'uk': 'united kingdom',
        'u.s.a.': 'united states',
        'u.s.': 'united states',
        'u.k.': 'united kingdom',
        'great britain': 'united kingdom',
        'holland': 'netherlands',
        'the netherlands': 'netherlands',
        'switzerland': 'swiss confederation',
        'vatican': 'holy see',
        'vatican city': 'holy see',
        'taiwan': 'taiwan, province of china',
        'hong kong': 'hong kong, special administrative region of china',
        'macau': 'macao, special administrative region of china',
    }
    
    # Check if the country name is in our mappings
    if country_name in country_mappings:
        return country_mappings[country_name]
    
    # Remove common prefixes/suffixes that might interfere with matching
    country_name = re.sub(r'^the\s+', '', country_name)
    country_name = re.sub(r'\s+\(.*?\)$', '', country_name)
    
    return country_name

def derive_faker_locale(address_text: str) -> str:
    """
    Derives a valid Faker locale from a given address text.
    Process:
      1. Extract the country from text using spaCy NER.
      2. Retrieve language info via CountryInfo.
      3. Convert the language to an ISO code using pycountry.
      4. Get the country's alpha_2 code using pycountry.
      5. Optionally adjust the language based on direct language detection.
      
    Returns a locale string like 'es_ES' or falls back to 'en_US'.
    """
    print(f"\n[DEBUG] derive_faker_locale called with input: '{address_text[:100]}...'")
    
    if not address_text or not isinstance(address_text, str):
        print("[DEBUG] Empty or invalid input, returning en_US")
        return "en_US"
    
    # 1. Extract country name from the address text using cached spaCy model
    try:
        print("[DEBUG] Loading spaCy model...")
        nlp = get_nlp()
        print("[DEBUG] Processing text with spaCy...")
        doc = nlp(address_text)
        print(f"[DEBUG] Found {len(doc.ents)} entities in text")
        
        # Print all entities for debugging
        for ent in doc.ents:
            print(f"[DEBUG] Entity: '{ent.text}', Label: {ent.label_}")
        
        country_name = next((ent.text for ent in doc.ents if ent.label_ == "GPE"), None)
        print(f"[DEBUG] Extracted country name from NER: '{country_name}'")
        
        # If no country found, try to extract from common patterns
        if not country_name:
            print("[DEBUG] No country found via NER, trying regex patterns...")
            # Look for common country indicators in the text
            country_patterns = [
                r'country:\s*([^,\.]+)',
                r'country\s*=\s*([^,\.]+)',
                r'country\s*:\s*([^,\.]+)',
                r'country\s*=\s*([^,\.]+)',
                r'([A-Z]{2})\s*$',  # Two-letter country code at the end
            ]
            
            for pattern in country_patterns:
                print(f"[DEBUG] Trying pattern: {pattern}")
                match = re.search(pattern, address_text, re.IGNORECASE)
                if match:
                    country_name = match.group(1).strip()
                    print(f"[DEBUG] Found country via regex: '{country_name}'")
                    break
    except Exception as e:
        print(f"[DEBUG] ERROR extracting country from address: {str(e)}")
        logging.warning(f"Error extracting country from address: {str(e)}")
        country_name = None
    
    if not country_name:
        print("[DEBUG] No country name found, returning en_US")
        return "en_US"
    
    # Normalize the country name
    print(f"[DEBUG] Normalizing country name: '{country_name}'")
    normalized_country = normalize_country_name(country_name)
    print(f"[DEBUG] Normalized country name: '{normalized_country}'")
    
    # 2. Get country info and derive its primary language
    try:
        print(f"[DEBUG] Getting country info for: '{normalized_country}'")
        info = CountryInfo(normalized_country).info()
        languages = info.get("languages", [])
        print(f"[DEBUG] Languages found: {languages}")
        lang_name = languages[0] if languages else "English"
        print(f"[DEBUG] Selected language: '{lang_name}'")
    except Exception as e:
        print(f"[DEBUG] ERROR getting country info: {str(e)}")
        logging.warning(f"Error getting country info for {normalized_country}: {str(e)}")
        lang_name = "English"

    # 3. Convert language name to an ISO code
    try:
        print(f"[DEBUG] Converting language '{lang_name}' to ISO code")
        language_obj = pycountry.languages.get(name=lang_name)
        print(f"[DEBUG] Language object found: {language_obj is not None}")
        lang_code = language_obj.alpha_2 if language_obj and hasattr(language_obj, "alpha_2") else lang_name[:2].lower()
        print(f"[DEBUG] Language code: '{lang_code}'")
    except Exception as e:
        print(f"[DEBUG] ERROR converting language: {str(e)}")
        logging.warning(f"Error converting language {lang_name} to ISO code: {str(e)}")
        lang_code = lang_name[:2].lower() if len(lang_name) >= 2 else "en"

    # 4. Retrieve the country's alpha_2 code
    try:
        print(f"[DEBUG] Getting country code for: '{normalized_country}'")
        country_obj = pycountry.countries.get(name=normalized_country)
        print(f"[DEBUG] Country object found: {country_obj is not None}")
        
        if not country_obj:
            print("[DEBUG] Country not found directly, trying fuzzy match...")
            # As fallback, try to find a matching country from pycountry's list
            country_obj = next((c for c in pycountry.countries if normalized_country.lower() in c.name.lower()), None)
            print(f"[DEBUG] Fuzzy match found: {country_obj is not None}")
        
        country_code = country_obj.alpha_2 if country_obj and hasattr(country_obj, "alpha_2") else normalized_country[:2].upper()
        print(f"[DEBUG] Country code: '{country_code}'")
    except Exception as e:
        print(f"[DEBUG] ERROR getting country code: {str(e)}")
        logging.warning(f"Error getting country code for {normalized_country}: {str(e)}")
        country_code = normalized_country[:2].upper() if len(normalized_country) >= 2 else "US"

    # 5. Use direct language detection as a double-check
    try:
        print("[DEBUG] Attempting language detection...")
        detected_lang = detect(address_text)
        print(f"[DEBUG] Detected language: '{detected_lang}'")
        if lang_code.lower() == "en" and detected_lang != "en":
            print(f"[DEBUG] Overriding language code from '{lang_code}' to '{detected_lang}'")
            lang_code = detected_lang
    except LangDetectException as e:
        print(f"[DEBUG] LangDetectException: {str(e)}")
        # Specific exception for language detection failures
        pass
    except Exception as e:
        print(f"[DEBUG] ERROR in language detection: {str(e)}")
        logging.warning(f"Error in language detection: {str(e)}")
        pass

    # Construct the locale string
    locale_str = f"{lang_code.lower()}_{country_code.upper()}"
    print(f"[DEBUG] Constructed locale string: '{locale_str}'")
    
    # Check if the locale is in the problematic list
    if locale_str in PROBLEMATIC_LOCALES:
        print(f"[DEBUG] Problematic locale detected: '{locale_str}'")
        logging.warning(f"Problematic locale detected: {locale_str}, falling back to common locale")
        # Try to find a safer alternative for this language
        if lang_code.lower() in ['ar', 'zh', 'ja', 'ko', 'th']:
            fallback = common_locales.get(lang_code.lower(), "en_US")
            print(f"[DEBUG] Using fallback locale: '{fallback}'")
            return fallback
    
    # Fallback mapping for common languages
    common_locales = {
        'en': 'en_US', 'es': 'es_ES', 'fr': 'fr_FR', 'de': 'de_DE',
        'it': 'it_IT', 'pt': 'pt_BR', 'nl': 'nl_NL', 'ru': 'ru_RU',
        'ja': 'ja_JP', 'zh': 'zh_CN', 'ar': 'ar_SA'
    }

    # Check if the locale exists in Faker by trying to create a test instance
    print(f"[DEBUG] Testing if locale '{locale_str}' is valid...")
    try:
        test_faker = Faker(locale_str)
        print(f"[DEBUG] Locale '{locale_str}' is valid")
        return locale_str
    except Exception as e:
        print(f"[DEBUG] Locale '{locale_str}' is not valid, using fallback")
        # Use fallback locale based on language
        fallback = common_locales.get(lang_code.lower(), "en_US")
        print(f"[DEBUG] Using common locale fallback: '{fallback}'")
        return fallback

# For backwards compatibility, alias the function
get_faker_locale = derive_faker_locale

class DataAnonymizer:
    def __init__(self, openai_api_key: str):
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")
            
        self.client = OpenAI(api_key=openai_api_key)
        
        # Initialize faker with default locale
        self.faker = Faker()
        
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
        primary_path = Path(os.path.join(project_root, "analysis", "Prompt"))
        
        # Check the primary path first (from pasted text 1)
        if primary_path.exists():
            self.prompts_dir = primary_path
            self.logger.info(f"Found Prompt directory at primary location: {self.prompts_dir}")
        else:
            # Then try the alternative paths in order
            possible_paths = [
                Path("Prompt"),  # Current directory
                Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Prompt")),  # Same dir as this file
                Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Prompt")),  # One level up
                Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis", "Prompt")),  # In analysis folder
                Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Prompt"))  # Two levels up
            ]
            
            self.prompts_dir = None
            for path in possible_paths:
                if path.exists():
                    self.prompts_dir = path
                    self.logger.info(f"Found Prompt directory at alternative location: {self.prompts_dir}")
                    break
        
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
        
        # Cache for Faker instances to avoid recreating them
        self._faker_cache = {}

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
        
        # Special debug for address prompts
        if "[Limit Hierarchy]" in prompt:
            self.logger.error(f"ERROR: Unprocessed [Limit Hierarchy] found in prompt: {prompt[:200]}")
        
        if "Location Data" in prompt and "anonymize" in prompt.lower():
            self.logger.info(f"LOCATION PROMPT DETAILS:")
            self.logger.info(f"  - Prompt contains 'Location Data': YES")
            self.logger.info(f"  - Contains 'Level 1'? {'YES' if 'Level 1' in prompt else 'NO'}")
            self.logger.info(f"  - Contains 'Level 2'? {'YES' if 'Level 2' in prompt else 'NO'}")
            self.logger.info(f"  - Contains 'Level 3'? {'YES' if 'Level 3' in prompt else 'NO'}")
            self.logger.info(f"  - Contains 'Level 4'? {'YES' if 'Level 4' in prompt else 'NO'}")
            self.logger.info(f"  - Contains 'Level 5'? {'YES' if 'Level 5' in prompt else 'NO'}")
        
        # Log the prompt being sent (only once)
        self.logger.info(f"Sending prompt to GPT:\n{prompt}\n{'='*50}")
      
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
                
                # Log the response
                self.logger.info(f"GPT Response:\n{result}\n{'='*50}")
                
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

    def create_anonymization_prompt(self, value: Any, field_type: str, field_name: str, address_dict: Optional[dict] = None) -> str:
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
            prompt = prompt.replace("[Job Title]", value_str)
        elif "[Employer]" in prompt:
            prompt = prompt.replace("[Employer]", value_str)
        elif field_type == "address" and "[Location Data]" in prompt:
            # First assemble the full address string from all available components
            
            # Start with the value that was passed to the function
            address_string = value_str
            print(f"\n[DEBUG] Initial address string: '{address_string}'")
            
            # Include full address details if available
            if address_dict:
                # Process the dynamic address object
                print(f"[DEBUG] Processing address dictionary with {len(address_dict)} fields: {list(address_dict.keys())}")
                self.logger.info(f"Processing address dictionary with {len(address_dict)} fields: {list(address_dict.keys())}")
                
                # Extract the location level (should be the last field)
                level = address_dict.get("level", "Level 3")
                print(f"[DEBUG] Location level: '{level}'")
                
                # Build a combined address string from all fields except 'level'
                address_components = []
                for key, val in address_dict.items():
                    if key != 'level' and val and isinstance(val, str):
                        # Include the field name and value
                        address_components.append(f"{key}: {val}")
                
                # Join all components into a single string
                if address_components:
                    address_string = ", ".join(address_components)
                    print(f"[DEBUG] Assembled address string: '{address_string}'")
                    self.logger.info(f"Assembled address string: {address_string}")
            
            # Now that we have the complete address, detect locale and initialize Faker
            print("[DEBUG] Calling get_faker_locale to detect locale...")
            locale = get_faker_locale(address_string)
            print(f"[DEBUG] Detected locale: '{locale}'")
            self.logger.info(f"Address locale detection - Input address: '{address_string[:100]}...', Detected locale: '{locale}'")
            
            # Initialize Faker with the detected locale, with retry logic and caching
            fake_address = None
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries and not fake_address:
                try:
                    print(f"[DEBUG] Attempt {retry_count+1}/{max_retries} to generate fake address with locale '{locale}'")
                    # Check if we already have a Faker instance for this locale
                    if locale in self._faker_cache:
                        print(f"[DEBUG] Using cached Faker instance for locale '{locale}'")
                        self.faker = self._faker_cache[locale]
                    else:
                        print(f"[DEBUG] Creating new Faker instance for locale '{locale}'")
                        self.faker = Faker(locale)
                        self._faker_cache[locale] = self.faker
                    
                    print("[DEBUG] Generating street address...")
                    fake_address = self.faker.street_address()
                    print(f"[DEBUG] Generated fake address: '{fake_address}'")
                    self.logger.info(f"Generated fake address: '{fake_address}'")
                except Exception as e:
                    retry_count += 1
                    print(f"[DEBUG] ERROR generating fake address (attempt {retry_count}/{max_retries}): {str(e)}")
                    self.logger.warning(f"Error using locale '{locale}' (attempt {retry_count}/{max_retries}): {str(e)}")
                    if retry_count >= max_retries:
                        print("[DEBUG] Max retries reached, falling back to en_US")
                        self.logger.warning(f"Falling back to en_US after {max_retries} attempts")
                        self.faker = Faker('en_US')
                        fake_address = self.faker.street_address()
                    else:
                        # Try a different locale based on the language code
                        lang_code = locale.split('_')[0]
                        fallback_locale = f"{lang_code}_{lang_code.upper()}"
                        print(f"[DEBUG] Trying alternative locale: '{fallback_locale}'")
                        if fallback_locale in Faker.locales and fallback_locale != locale:
                            locale = fallback_locale
                            self.logger.info(f"Trying alternative locale: {locale}")
                        else:
                            # Wait before retrying
                            print(f"[DEBUG] Waiting 1 second before retry...")
                            time.sleep(1)
            
            # Now replace the placeholders in the prompt
            print("[DEBUG] Replacing placeholders in prompt")
            prompt = prompt.replace("[Location Data]", address_string)
            prompt = prompt.replace("[Fake Address]", fake_address)
                
            # Make sure the level is properly set in the prompt
            if "[Limit Hierarchy]" in prompt:
                print(f"[DEBUG] Setting location anonymization level to: '{level}'")
                self.logger.info(f"Setting location anonymization level to: {level}")
                prompt = prompt.replace("[Limit Hierarchy]", level)
                        
            prompt = prompt.replace("[Dimensions]", str(len(address_string.split(','))))
            
            # Log that we're handling an address field
            print(f"[DEBUG] Processing address field with value: '{value_str[:100] if len(value_str) > 100 else value_str}'")
            self.logger.info(f"Processing address field with value: {value_str[:100] if len(value_str) > 100 else value_str}")
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
            
        # Special handling for address dictionaries
        address_dict = None
        original_structure = None
        
        if field_type == "address" and isinstance(value, dict):
            # Save the original address dictionary and structure
            address_dict = value
            original_structure = list(value.keys())
            self.logger.info(f"Original address structure: {original_structure}")
            
            # Create a combined address string for the main anonymization
            address_components = []
            for key, val in value.items():
                if key != 'level' and val and isinstance(val, str):
                    # Include the field name and value
                    address_components.append(f"{key}: {val}")
            
            # Join all components into a single string
            if address_components:
                value_for_prompt = ", ".join(address_components)
                self.logger.info(f"Combined address for prompt: {value_for_prompt}")
            else:
                # Fallback to JSON string if no components found
                value_for_prompt = json.dumps(value)
        else:
            # For non-address fields or non-dict values
            value_for_prompt = value
            
        if not isinstance(value_for_prompt, (str, int, float, bool)):
            self.logger.warning(f"Converting {field_name} of type {type(value_for_prompt)} to string")
            value_for_prompt = str(value_for_prompt)
        
        try:
            # Create appropriate prompt from the text file
            prompt = self.create_anonymization_prompt(value_for_prompt, field_type, field_name, address_dict)
            
            self.logger.info(f"AFTER PROMPT CREATION - Field: {field_name}, Prompt length: {len(prompt)}")
            self.logger.debug(f"Generated prompt for {field_name} (truncated): {prompt[:100]}...")
            
            anonymized_value = self.get_gpt_response(prompt)
            
            # Clean up the response
            cleaned_value = self._clean_gpt_response(anonymized_value, field_type)
            
            # For address fields, try to maintain the original structure
            if field_type == "address" and isinstance(cleaned_value, dict) and original_structure:
                # Check if we have a single address string in the response
                self.logger.info(f"Cleaned address response: {cleaned_value}")
                
                # Reconstruct result to match the original structure
                result = {}
                
                # If we have a structured result with multiple fields
                if len(cleaned_value.keys()) > 1 and not (len(cleaned_value.keys()) == 2 and 'level' in cleaned_value):
                    # Keep the structured format but enforce original keys where possible
                    for orig_key in original_structure:
                        if orig_key == 'level':
                            # Preserve the level setting
                            result[orig_key] = address_dict.get('level', 'Level 3')
                            continue
                            
                        # Try to find a matching key in the anonymized result
                        matching_key = None
                        for key in cleaned_value.keys():
                            if key.lower() == orig_key.lower() or key.lower().replace(' ', '_') == orig_key.lower():
                                matching_key = key
                                break
                                
                        if matching_key:
                            # Use the original key with the anonymized value
                            result[orig_key] = cleaned_value[matching_key]
                        else:
                            # If no direct match, keep the original key but set empty value
                            result[orig_key] = ""
                else:
                    # We got back a simple format or single value - distribute to original structure
                    # Get the primary address value 
                    primary_value = None
                    for key, value in cleaned_value.items():
                        if key != 'level' and isinstance(value, str) and value.strip():
                            primary_value = value
                            break
                            
                    if not primary_value:
                        # If no primary value found, convert the whole dict to a string
                        primary_value = str(cleaned_value)
                        
                    # Find the main address field in the original structure
                    address_key = next((k for k in original_structure 
                                     if 'address' in k.lower() or 'street' in k.lower()), original_structure[0])
                                     
                    # Restore the original structure with the anonymized primary value
                    for key in original_structure:
                        if key == 'level':
                            result[key] = address_dict.get('level', 'Level 3')
                        elif key == address_key:
                            result[key] = primary_value
                        else:
                            result[key] = ""
                            
                self.logger.info(f"Reconstructed address result to match original structure: {result}")
                cleaned_value = result
            
            # Store in processed values for consistency within the current record
            if field_type == "name":
                self.processed_values["name"] = cleaned_value
                
            return cleaned_value
            
        except Exception as e:
            self.logger.error(f"Error anonymizing {field_type} '{value}': {str(e)}")
            return value  # Return original value on error

    def anonymize_data(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Main anonymization function with input validation"""
        if not isinstance(input_data, dict):
            raise ValueError("Input data must be a dictionary")
            
        data = input_data.get("data", [])
        config = input_data.get("columnsConfig", {})
        
        if not isinstance(data, list):
            raise ValueError("Data must be a list of records")
            
        # Debug: Log categories to check for address fields
        address_fields = []
        for field_name, field_config in config.items():
            field_type = field_config.get("type", "none")
            if field_type == "address":
                address_fields.append(field_name)
        
        if address_fields:
            self.logger.info(f"Found address fields in config: {address_fields}")
        else:
            self.logger.warning("No address fields found in config!")
            
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
            
            # Log address objects in the record for debugging
            if 'address' in record:
                self.logger.info(f"Record {i} has address object: {type(record['address'])}")
                if isinstance(record['address'], dict):
                    self.logger.info(f"Address object contents: {json.dumps(record['address'], indent=2)}")
                    if 'level' in record['address']:
                        self.logger.info(f"Address level found: {record['address']['level']}")
            
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

