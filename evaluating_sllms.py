import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from bge_evaluation import *
from tqdm import tqdm
import pickle
import argparse

def process_math_shepherd():
    dataset = load_dataset('peiyi9979/Math-Shepherd')['train'].to_pandas()
    inputs = dataset['input'].apply(lambda x: x[:x.find("Step 1")].strip()).to_list() # list of questions
    outputs = dataset['input'].apply(lambda x: x[x.find("Step 1"):].strip()).to_list() # list of list of steps
    return inputs, outputs

def main(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    # model = AutoModelForCausalLM.from_pretrained(model_name)#, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to('cpu')
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model).module
    
    ## left padding for generation
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    instruction_prompt = lambda problem: f"Solve the following problem, and break down your solution into a set of steps:\n{problem}"
    
    def perform_inference(prompts, references, batch_size=98):
        tokenizer.model_max_length = 1024
        max_length = int(tokenizer.model_max_length)

        all_metrics = []
        all_outputs = []

        # Process prompts in batches
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i+batch_size]
            batch_references = references[i:i+batch_size]

            # Modify prompts in place
            for j, prompt in enumerate(batch_prompts):
                batch_prompts[j] = instruction_prompt(prompt)

            tokenized_output = tokenizer(
                list(batch_prompts), 
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(model.device)

            with torch.no_grad():
                gen_tokens = model.generate(
                    **tokenized_output,
                    max_new_tokens=150,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    min_p=0.1,
                    temperature=0.2,
                )

            # Extract only the new tokens
            new_tokens = gen_tokens[:, tokenized_output.input_ids.shape[1]:]
            
            # Decode only the new tokens
            batch_outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            batch_performance = evaluate_bge(batch_outputs, batch_references)

            all_metrics.extend(batch_performance)
            all_outputs.extend(batch_outputs)

        return all_outputs, np.array(all_metrics)

    ms_input, ms_output = process_math_shepherd()
    ms_generated, ms_performance = perform_inference(ms_input, ms_output)

    model_suffix = model_name[model_name.rfind('/')+1:].replace('_', '-')
    with open(f"{model_suffix}_math-shepherd.pkl", 'wb+') as f:
        pickle.dump([ms_input, ms_output, ms_generated, ms_performance])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    args = parser.parse_args()

    main(args.model_name)