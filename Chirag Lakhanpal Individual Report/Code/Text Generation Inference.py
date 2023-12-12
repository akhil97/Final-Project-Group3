import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
import argparse
from huggingface_hub import hf_hub_download
import shutil


def generate_legal_document(prompt, model, tokenizer, max_length=512, max_new_tokens=50):
    """
    Generate a legal document based on the given prompt.

    Parameters:
    prompt (str): The starting text of the document.
    model: The pre-trained GPT-2 model.
    tokenizer: The GPT-2 tokenizer.
    max_length (int): The maximum total length of the input sequence.
    max_new_tokens (int): The maximum length of the new tokens generated.

    Returns:
    str: The generated legal document.
    """

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    encoding = tokenizer.encode_plus(
        prompt, 
        return_tensors='pt',
        add_special_tokens=True, 
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True
    )
    input_ids = encoding['input_ids'].to(model.device)
    attention_mask = encoding['attention_mask'].to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids, 
            attention_mask=attention_mask,
            max_length=max_length + max_new_tokens, 
            num_beams=5, 
            no_repeat_ngram_size=2, 
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def download_model_from_hub(repo_id, model_directory):
    """
    Download the model files from Hugging Face Hub and save them in the specified directory.

    Parameters:
    repo_id (str): Repository ID on Hugging Face Hub.
    model_directory (str): Directory where the model should be saved.
    """
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    file_names = ['config.json', 'generation_config.json', 'model.safetensors']

    for file_name in file_names:
        file_path = hf_hub_download(repo_id=repo_id, filename=f"{model_directory}/{file_name}")
        shutil.copy(file_path, os.path.join(model_directory, file_name))


def main():
    parser = argparse.ArgumentParser(description='Generate legal documents using GPT-2.')
    parser.add_argument('--model_path', type=str, default=None, 
                        help='Path to the pre-trained model. If not provided, the default model will be downloaded.')
    
    args = parser.parse_args()

    if args.model_path is None:
        model_directory = "fine_tuned_gpt2_E25_Large"
        repo_id = "chiraglakhanpal/Legal"
        print("Downloading the model from Hugging Face Hub...")
        download_model_from_hub(repo_id, model_directory)
        model_path = model_directory
    else:
        model_path = args.model_path

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prompt = input("Enter a prompt: ")

    generated_document = generate_legal_document(prompt, model, tokenizer, max_length=512, max_new_tokens=512)
    print(generated_document)
    
if __name__ == "__main__":
    main()
