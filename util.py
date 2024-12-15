from groq import Groq
import base64
import dotenv
import os
import re
import io

dotenv.load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def predict_class_llama(image, model_name="llama-3.2-11b-vision-preview"):
    try:
        image_byte_arr = io.BytesIO()
        image.save(image_byte_arr, format='JPEG')
        image_byte_arr = image_byte_arr.getvalue()

        # Codifica os bytes em Base64
        base64_image = base64.b64encode(image_byte_arr).decode('utf-8')

        # Encode the image
        if not base64_image:
            return None
        # Load API key from environment variables
        dotenv.load_dotenv()
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        # Initialize Groq client
        client = Groq(api_key=GROQ_API_KEY)

        # Send the image and prompt to Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": 
                        """
                        Analyze the image and determine the material of the object by reasoning through its visible features (such as texture, color, shape, context, or other relevant details). 

                        The material must belong to one of the following 8 categories: ['Batterie', 'Carton', 'Metal', 'Organique', 'Papier', 'Plastique', 'Verre', 'Vetements'].

                        Provide a detailed reasoning for your classification, explaining why the object fits the chosen category and why it does not belong to others, if applicable.

                        Conclude your response with a clear identification of the selected class in the following format:
                        - Identified as being 'CLASS'

                        Finally, include the class in a JSON-like output as follows:
                        {'class': 'CLASS'}
                        """},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model= model_name,
        )

        # Extract the response text
        text = chat_completion.choices[0].message.content

        # Use regex to extract the predicted class
        match = re.search(r"[{]\s*[\"']class[\"']\s*:\s*[\"']([^\"']+)[\"']\s*[}]", text)
        # print(text)
        if match:
            return match.group(1)  # Return only the class name
        else:
            print("Class not found in the response")
            return "Inconnu"

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Inconnu"