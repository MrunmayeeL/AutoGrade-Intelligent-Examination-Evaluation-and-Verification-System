import os
import sys
import json
import argparse
from ocr_pixtral import client, image_to_base64

def generate_answer_key_from_image(image_path):
    print(f"Reading image: {image_path}...")
    if not os.path.exists(image_path):
        print(f"Error: Image path '{image_path}' does not exist.")
        sys.exit(1)
        
    image_base64 = image_to_base64(image_path)
    
    prompt = """
    Analyze this answer key image. Extract all questions, their ideal reference answers, and the marks allotted to each question.
    
    Format the extracted information into a clean JSON structure where each key is the question code (e.g. "Q1", "Q2") and the value is an object containing:
    1. "answer": The ideal reference answer text.
    2. "max_marks": The maximum marks allotted for this question (integer). If not explicitly visible, default to 5 marks.
    
    Make sure to return ONLY a valid JSON object matching this schema. Do not add markdown formatting outside the JSON, and do not add explanations.
    
    Example Schema:
    {
      "Q1": {
        "answer": "Ideal answer text...",
        "max_marks": 7
      },
      "Q2": {
        "answer": "Ideal answer text...",
        "max_marks": 10
      }
    }
    """
    
    print("Sending image to Pixtral for structured parsing...")
    try:
        response = client.chat.completions.create(
            model="pixtral-large-latest",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=3000,
            temperature=0.1
        )
        
        raw_json = response.choices[0].message.content.strip()
        data = json.loads(raw_json)
        return data
    except Exception as e:
        print(f"Error calling Pixtral API: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Convert an answer key image into structured JSON format.")
    parser.add_argument("image", help="Path to the answer key image file.")
    parser.add_argument("--output", default="answer_key.json", help="Path to save the generated JSON file.")
    
    args = parser.parse_args()
    
    data = generate_answer_key_from_image(args.image)
    
    # Save to file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        
    print(f"\nSuccess! Structured answer key saved to: {args.output}")

if __name__ == "__main__":
    main()
