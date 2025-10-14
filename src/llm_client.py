import os
import logging
from time import sleep
from typing import Optional
import google.genai as genai
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClientGemini:
    """A wrapper class for handling Gemini LLM inference."""
    
    def __init__(self, 
                 model_checkpoint: str = 'gemini-2.5-pro',
                 max_tokens: int = 32768,
                 thinking_budget: int = 1000,
                 temperature: float = 0.6,
                 top_p: float = 0.95,
                 sleep_between_calls: float = 60,
                 api_key: Optional[str] = None,
                 debug: bool = False):
        """Initialize the Gemini LLM client."""
        self.model_checkpoint = model_checkpoint
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.top_p = top_p
        self.sleep_between_calls = sleep_between_calls
        self.debug = debug
        self._last_thought = None
        
        if not self.debug:
            if api_key is None:
                api_key = os.environ.get('GOOGLE_API_KEY')
            assert api_key is not None, "Missing GOOGLE_API_KEY"
            
            self.api_key = api_key
            self.client, self.generation_config = self._configure_client()
        else:
            self.client = None

        logger.info(f"Gemini LLM Client initialized with model: {self.model_checkpoint} (debug={self.debug})")

    def _configure_client(self):
        """Set up and return the LLM with the provided configuration."""
        generation_config = genai.types.GenerateContentConfig(
            top_k=50,
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            thinking_config=genai.types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
                include_thoughts=self.thinking_budget > 0
            )
        )
        client = genai.Client(api_key=self.api_key)
        return client, generation_config
    
    def generate(self, prompt: str, do_sleep: bool = True, **kwargs) -> str:
        """Generate a response from the Gemini LLM."""
        if self.debug:
            logger.info("Debug mode: returning dummy response")
            return "Dummy content for testing pipeline functionality."

        logger.info("Running Gemini inference...")
        
        response = self.client.models.generate_content(
            model=self.model_checkpoint,
            contents=prompt,
            config=self.generation_config
        )
        
        thought = None
        response_text = None
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                thought = part.text.strip()
            else:
                response_text = part.text.strip()
                logger.info(f"LLM response: {response_text}")
        
        self._last_thought = thought
        
        if do_sleep and self.sleep_between_calls > 0:
            logger.info(f"Sleeping for: {self.sleep_between_calls}s")
            sleep(self.sleep_between_calls)
            
        return response_text
    
    def generate_and_extract(self, prompt: str, do_sleep: bool = True, **kwargs) -> tuple[str, str]:
        """Generate a response and return both full response and extracted content."""
        response = self.generate(prompt, do_sleep, **kwargs)
        thought = self._last_thought or ""
        
        if self.debug:
            full_response = response
            extracted_content = response
        else:
            full_response = f"<think>\n{thought}\n</think>\n\n{response}" if thought else response
            extracted_content = response
        
        return full_response, extracted_content


class LLMClientDeepseek:
    """A wrapper class for handling DeepSeek LLM inference via Together API."""
    
    def __init__(self, 
                 model_checkpoint: str = 'deepseek-ai/DeepSeek-R1',
                 max_tokens: int = 32768,
                 temperature: float = 0.6,
                 top_p: float = 0.95,
                 sleep_between_calls: float = 60,
                 api_key: Optional[str] = None,
                 debug: bool = False):
        """Initialize the DeepSeek LLM client."""
        self.model_checkpoint = model_checkpoint
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.sleep_between_calls = sleep_between_calls
        self.debug = debug
        
        if not self.debug:
            if api_key is None:
                api_key = os.environ.get('TOGETHER_API_KEY')
            assert api_key is not None, "Missing TOGETHER_API_KEY"
            
            try:
                from together import Together
                self.client = Together(api_key=api_key)
            except ImportError:
                raise ImportError("Please install the 'together' package: pip install together")
        else:
            self.client = None

        logger.info(f"DeepSeek LLM Client initialized with model: {self.model_checkpoint} (debug={self.debug})")
    
    def generate(self, prompt: str, do_sleep: bool = True, **kwargs) -> str:
        """Generate a response from the DeepSeek LLM."""
        if self.debug:
            logger.info("Debug mode: returning dummy response")
            return "<think>\nThis is a dummy response for debugging.\n</think>\n\nDummy content for testing."

        logger.info("Running DeepSeek inference...")
        
        response = self.client.chat.completions.create(
            model=self.model_checkpoint,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        
        response_text = response.choices[0].message.content
        logger.info(f"LLM response: {response_text}")
        
        if do_sleep and self.sleep_between_calls > 0:
            logger.info(f"Sleeping for: {self.sleep_between_calls}s")
            sleep(self.sleep_between_calls)
            
        return response_text
    
    def extract_content_after_think(self, response: str) -> str:
        """Extract content after the </think> tag."""
        return response.split('</think>')[-1].strip()
    
    def generate_and_extract(self, prompt: str, do_sleep: bool = True, **kwargs) -> tuple[str, str]:
        """Generate a response and extract content after </think> tag."""
        full_response = self.generate(prompt, do_sleep, **kwargs)
        extracted_content = self.extract_content_after_think(full_response)
        return full_response, extracted_content


class PromptManager:
    """Helper class for managing prompts and templates."""
    
    @staticmethod
    def load_prompt(prompt_path: str) -> str:
        """Load a prompt from file."""
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def load_prompts(prompt_dir: str, prompt_files: list[str]) -> dict[str, str]:
        """Load multiple prompts from a directory."""
        prompts = {}
        for filename in prompt_files:
            filepath = os.path.join(prompt_dir, filename)
            key = os.path.splitext(filename)[0]  # Remove extension for key
            prompts[key] = PromptManager.load_prompt(filepath)
        return prompts
    
    @staticmethod
    def format_prompt(template: str, **kwargs) -> str:
        """Format a prompt template with the given kwargs."""
        return template.format(**kwargs)