"""Document Retrieval module."""

import numpy as np
from sentence_transformers import SentenceTransformer

def cosine_similarity(v1, v2):
    """
    Compute the cosine similarity between two vectors.

    Parameters:
    v1 (numpy.ndarray): The first vector.
    v2 (numpy.ndarray): The second vector.

    Returns:
    float: The cosine similarity between the two vectors.
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

def vector_space_model(query, documents, k=5):
    """
    Calculates the top k similar documents to a given query using the Vector Space Model.

    Args:
        query (str): The query string.
        documents (list): A list of document strings.
        k (int, optional): The number of similar documents to return. Defaults to 5.

    Returns:
        list: A list of tuples containing the top k similar documents and their similarity scores.
    """
    # Tokenize query and documents
    query_tokens = query.lower().split()
    document_tokens = [doc.lower().split() for doc in documents]

    # Create vocabulary
    vocabulary = list(set(query_tokens))
    for doc_tokens in document_tokens:
        vocabulary.extend(doc_tokens)
    vocabulary = list(set(vocabulary))

    # Create term frequency (TF) matrix
    tf_matrix = np.zeros((len(documents), len(vocabulary)))
    for i, doc_tokens in enumerate(document_tokens):
        for token in doc_tokens:
            tf_matrix[i, vocabulary.index(token)] += 1

    # Create document frequency (DF) vector
    df_vector = np.zeros(len(vocabulary))
    for token in query_tokens:
        df_vector[vocabulary.index(token)] += 1

    # Calculate inverse document frequency (IDF) vector
    idf_vector = np.log(len(documents) / (df_vector + 1))

    # Calculate TF-IDF matrix
    tfidf_matrix = tf_matrix * idf_vector

    # Calculate query vector
    query_vector = np.zeros(len(vocabulary))
    for token in query_tokens:
        if token in vocabulary:
            query_vector[vocabulary.index(token)] += 1
    query_vector *= idf_vector

    # Calculate cosine similarity between query vector and document vectors
    similarities = []
    for i in range(len(documents)):
        sim = cosine_similarity(query_vector, tfidf_matrix[i])
        similarities.append((i, sim))

    # Sort documents by similarity score
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top k similar documents
    top_k_similar_docs = []
    for i in range(min(k, len(similarities))):
        doc_index = similarities[i][0]
        top_k_similar_docs.append((documents[doc_index], similarities[i][1]))

    return top_k_similar_docs


def bm25(query, documents, k=5, b=0.75, k1=1.5):
    """
    Implement BM25 algorithm to retrieve top k similar documents to a query.

    Parameters:
    query (str): The query string.
    documents (list): A list of document strings.
    k (int, optional): The number of top similar documents to retrieve. Defaults to 5.
    b (float, optional): The parameter controlling the impact of document length on BM25 score. Defaults to 0.75.
    k1 (float, optional): The parameter controlling the impact of term frequency on BM25 score. Defaults to 1.5.

    Returns:
    list: A list of tuples containing the top k similar documents and their corresponding BM25 scores.
    """
    # Tokenize query and documents
    query_tokens = query.lower().split()
    document_tokens = [doc.lower().split() for doc in documents]

    # Create vocabulary
    vocabulary = list(set(query_tokens))
    for doc_tokens in document_tokens:
        vocabulary.extend(doc_tokens)
    vocabulary = list(set(vocabulary))

    # Calculate document lengths
    doc_lengths = np.array([len(doc_tokens) for doc_tokens in document_tokens])

    # Create term frequency (TF) matrix
    tf_matrix = np.zeros((len(documents), len(vocabulary)))
    for i, doc_tokens in enumerate(document_tokens):
        for token in doc_tokens:
            tf_matrix[i, vocabulary.index(token)] += 1

    # Calculate document frequency (DF) vector
    df_vector = np.zeros(len(vocabulary))
    for token in query_tokens:
        df_vector[vocabulary.index(token)] += 1

    # Calculate inverse document frequency (IDF) vector
    idf_vector = np.log((len(documents) - df_vector + 0.5) / (df_vector + 0.5))

    # Calculate average document length
    avg_doc_length = np.mean(doc_lengths)

    # Calculate BM25 scores
    scores = []
    for i in range(len(documents)):
        tf = tf_matrix[i]
        doc_length = doc_lengths[i]
        score = np.sum(idf_vector * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_length / avg_doc_length)))
        scores.append((i, score))

    # Sort documents by BM25 score
    scores.sort(key=lambda x: x[1], reverse=True)

    # Return top k similar documents
    top_k_similar_docs = []
    for i in range(min(k, len(scores))):
        doc_index = scores[i][0]
        top_k_similar_docs.append((documents[doc_index], scores[i][1]))

    return top_k_similar_docs


def sentence_transformers_retrieval(query, documents, k=5, model_name='paraphrase-MiniLM-L6-v2'):
    """
    Apply sentence transformers to retrieve top k similar documents.

    Args:
        query (text): A search query for a document.
        documents (list): A list of documents for the search.
        k (int, optional): Number of top results to return. Defaults to 5.
        model_name (str, optional): Open source sentence model to use. Defaults to 'paraphrase-MiniLM-L6-v2'.

    Returns:
        list: A list of documents relevant to the search query.
    """
    # Load pre-trained sentence transformer model. Specify the model using model_name
    model = SentenceTransformer(model_name)

    # Encode query and documents into embeddings
    query_embedding = model.encode([query])[0]
    document_embeddings = model.encode(documents)

    # Calculate cosine similarity between query and documents
    similarities = [cosine_similarity(query_embedding, doc_embedding) for doc_embedding in document_embeddings]

    # Sort documents by similarity score
    similarities_with_indices = list(enumerate(similarities))
    similarities_with_indices.sort(key=lambda x: x[1], reverse=True)

    # Return top k similar documents
    top_k_similar_docs = []
    for i in range(min(k, len(similarities_with_indices))):
        doc_index = similarities_with_indices[i][0]
        top_k_similar_docs.append((documents[doc_index], similarities_with_indices[i][1]))

    return top_k_similar_docs
