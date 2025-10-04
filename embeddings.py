import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import gensim.downloader as api

def tweets_into_embeddings(texts, embeddingType="bert", batch_size=32):
    """
    Convert a list of texts into embeddings (BERT or Word2Vec).

    Parameters:
        texts (list of str): list of text strings
        embeddingType (str): "bert" or "word2vec"
        batch_size (int): batch size for BERT embeddings

    Returns:
        np.ndarray: shape (num_texts, embedding_size)
    """

    text_embeddings = []

    if embeddingType.lower() == "bert":
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model_bert = AutoModel.from_pretrained("distilbert-base-uncased")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            tokens = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            
            with torch.no_grad():
                outputs = model_bert(**tokens)
                cls_batch = outputs.last_hidden_state[:,0,:].numpy()
            
            text_embeddings.append(cls_batch)
        
        text_embeddings = np.vstack(text_embeddings)

    elif embeddingType.lower() == "word2vec":
        word2vec_model = api.load("word2vec-google-news-300")
        
        for text in texts:
            tokens = text.split()
            vectors = [word2vec_model[token] for token in tokens if token in word2vec_model]
            
            if len(vectors) == 0:
                vec = np.zeros(word2vec_model.vector_size)
            else:
                vec = np.mean(vectors, axis=0)
            
            text_embeddings.append(vec)
        
        text_embeddings = np.vstack(text_embeddings)
    
    else:
        raise ValueError("embeddingType must be 'bert' or 'word2vec'")
    
    return text_embeddings
