import argparse
import requests
import sys
import os
import re

# --- Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"
OUTPUT_DIR = "output"
 
# --- Main Functionality ---

def get_response_from_ollama(prompt):
    """
    Sends a prompt to the local Ollama server and returns the generated response.
    
    Args:
        prompt (str): The full prompt to send to the model, including context.
        
    Returns:
        str: The model's generated response text, or an error message.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False # We want the full response at once
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        # Ollama's /api/generate returns a JSON object. We need to extract the 'response' key.
        response_json = response.json()
        return response_json.get("response", "No response found in the API output.")
        
    except requests.exceptions.RequestException as e:
        return f"An error occurred while connecting to Ollama: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def generate_prompt(file_path, user_query):
    """
    Constructs a detailed prompt for the LLM by combining the system prompt,
    the content of the file, and the user's query.
    
    Args:
        file_path (str): The path to the code file.
        user_query (str): The question from the user.
        
    Returns:
        str: The complete prompt string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
            
        # The system prompt sets the context and persona for the LLM.
        system_prompt = (
            "You are a helpful and knowledgeable code assistant. "
            "Your task is to analyze the provided code and answer the user's question. "
            "Be concise and clear in your explanation, and provide code examples if they are relevant to the user's query."
        )
        
        # Combine everything into a single prompt for the model.
        full_prompt = (
            f"{system_prompt}\n\n"
            f"--- CODE FILE: {file_path} ---\n"
            f"```\n{code_content}\n```\n\n"
            f"--- USER QUESTION ---\n"
            f"{user_query}"
        )
        
        return full_prompt
        
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.", file=sys.stderr)
        sys.exit(1)
    except IOError:
        print(f"Error: Could not read the file '{file_path}'.", file=sys.stderr)
        sys.exit(1)

def save_response_to_file(file_path, user_query, response_text):
    """
    Saves the assistant's response to a new markdown file with an incrementing name
    in the format 'out-XXXXX.md'.
    
    Args:
        file_path (str): The path to the code file that was analyzed.
        user_query (str): The user's original query.
        response_text (str): The response from the LLM.
    """
    # Ensure the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
        except OSError as e:
            print(f"Error: Could not create output directory '{OUTPUT_DIR}': {e}", file=sys.stderr)
            return

    # Find the next available file number
    pattern = re.compile(r"out-(\d{5})\.md")
    max_num = -1
    try:
        for filename in os.listdir(OUTPUT_DIR):
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
    except OSError as e:
        print(f"Error: Could not read output directory '{OUTPUT_DIR}': {e}", file=sys.stderr)
        return

    next_num = max_num + 1
    output_filepath = os.path.join(OUTPUT_DIR, f"out-{next_num:05d}.md")

    # Format the content to be saved for better context
    output_content = f"# Analysis of: `{os.path.basename(file_path)}`\n\n**Query:**\n> {user_query}\n\n---\n\n{response_text}"

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(output_content)
        print(f"Response saved to '{output_filepath}'")
    except IOError as e:
        print(f"Error: Could not save response to '{output_filepath}': {e}", file=sys.stderr)

def main():
    """
    Main function to parse arguments and run the code assistant.
    """
    parser = argparse.ArgumentParser(
        description="A command-line code assistant using Ollama.",
        epilog="Example: python code_assistant.py my_script.py 'What does the main function do?'"
    )
    
    parser.add_argument(
        'file',
        type=str,
        help='The path to the code file to analyze.'
    )
    
    parser.add_argument(
        'query',
        type=str,
        help='The question you have about the code.'
    )
    
    args = parser.parse_args()
    
    file_path = args.file
    user_query = args.query
    
    print(f"Analyzing '{file_path}' with query: '{user_query}'...")
    print("Please wait for the response from Ollama...")
    
    # Generate the prompt and get the response
    prompt = generate_prompt(file_path, user_query)
    response_text = get_response_from_ollama(prompt)
    
    # Print the final response
    print("\n--- ASSISTANT RESPONSE ---")
    print(response_text)
    print("--------------------------\n")

    # Save the response to a file
    save_response_to_file(file_path, user_query, response_text)

if __name__ == "__main__":
    main()
