from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Example Pydantic model for response validation
class CustomModel(BaseModel):
    role: str
    content: str

def structured_generator(openai_model, prompt, custom_model):
    # Call OpenAI API to generate structured response
    result = openai_client.chat.completions.create(
        model=openai_model,
        response_model=custom_model.dict(),
        messages=[{"role": "user", "content": f"{prompt}, output must be in json"}]
    )
    return result

# Example usage:
if __name__ == "__main__":
    prompt_text = "Example prompt"
    response_model = CustomModel(role="user", content="")

    response = structured_generator("gpt-3.5-turbo", prompt_text, response_model)
    print(response)
