from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch

embedding_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
embedding_model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5').to('cpu')
embedding_model.eval()

if embedding_tokenizer.pad_token is None:
    embedding_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    embedding_model.resize_token_embeddings(len(embedding_tokenizer))
    
embedding_model = embedding_model.to('cuda')

def find_embedding(prompt):
    encoded_input = embedding_tokenizer(prompt, padding=True, truncation=True, return_tensors='pt').to(embedding_model.device)
    with torch.no_grad():
        model_output = embedding_model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings).squeeze()
    return sentence_embeddings

def evaluate_bge(predictions, references):
    metrics = []

    for pred, ref in zip(predictions, references):
        pred_emb = find_embedding(pred)
        ref_emb = find_embedding(ref)
        metrics.append(ref_emb.dot(pred_emb).item())    
    return np.array(metrics)

def process(output):
    """
    Strips and returns only the first line (everything before the first \n).
    """

    output = output.strip()
    if '\n' in output:
        output = output[:output.find('\n')]
    if '</s>' in output:
        output = output[:output.find('</s>')]
    return output