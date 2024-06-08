from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import nltk
import re
import numpy as np
import networkx as nx
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance
from collections import Counter
from nltk.probability import FreqDist

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    sentences = sent_tokenize(text)
    return sentences

def sentence_similarity(sent1, sent2, stop_words=None):
    if stop_words is None:
        stop_words = []

    sent1 = [w.lower() for w in word_tokenize(sent1)]
    sent2 = [w.lower() for w in word_tokenize(sent2)]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        if w in stop_words:
            continue
        vector1[all_words.index(w)] += 1

    for w in sent2:
        if w in stop_words:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 != idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

def generate_summary(text, top_n=5):
    stop_words = stopwords.words('english')
    sentences = preprocess_text(text)

    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)

    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    ranked_sentence_indexes = [index for index, score in sorted(scores.items(), key=lambda item: -item[1])]

    summary = [sentences[idx] for idx in ranked_sentence_indexes[:top_n]]
    return ' '.join(summary)

def generate_topic(text):
    # Use the entire text for topic generation
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word.lower() not in stop_words]
    
    # Get the most common words, but filter by length to avoid too short or too long words
    fdist = FreqDist(filtered_words)
    common_words = [word for word, freq in fdist.most_common(10) if 3 <= len(word) <= 15]
    
    # Join the top words to form a topic
    topic = ' '.join(common_words[:5])
    
    return topic

@api_view(['POST'])
def summarize(request):
    if request.method == 'POST':
        text = request.data.get('text', '')
        if not text:
            return Response({'error': 'Text field is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        summary = generate_summary(text)
        topic = generate_topic(text)
        
        return Response({'summary': summary, 'topic': topic }, status=status.HTTP_200_OK)
    return Response({'error': 'Invalid request'}, status=status.HTTP_400_BAD_REQUEST)
