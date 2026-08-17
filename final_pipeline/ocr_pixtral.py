import os
import base64
from openai import OpenAI

# Initialize client
client = OpenAI(
    api_key=os.getenv("MISTRAL_API_KEY"),
    base_url="https://api.mistral.ai/v1"
)

# Folders
INPUT_FOLDER = "handwritten_images"
OUTPUT_FOLDER = "extracted_text"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

def image_to_base64(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")

def extract_text_from_image(image_path):
    image_base64 = image_to_base64(image_path)

    response = client.chat.completions.create(
        model="pixtral-large-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract all handwritten text from this image. "
                            "Preserve line breaks and spacing as closely as possible. "
                            "Do not add explanations."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_base64}"
                    }
                ]
            }
        ],
        max_tokens=2000
    )

    return response.choices[0].message.content.strip()

def main():
    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith(SUPPORTED_EXTENSIONS):
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_filename = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            print(f"Processing: {filename}")

            try:
                extracted_text = extract_text_from_image(input_path)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(extracted_text)

                print(f"Saved → {output_filename}\n")

            except Exception as e:
                print(f"Failed on {filename}: {e}\n")

if __name__ == "__main__":
    main()
