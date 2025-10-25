"""
Text-based anomaly detectors adapted from LLM-Anomalous-Prompt-Detector repository.

This module integrates rule-based and NLP-based detection methods for identifying
anomalous prompts including:
- Rules and roleplay instruction combinations
- Color-changing code injections
- Length-based outliers
- Word frequency outliers
- Sentence semantic outliers
"""

import re
import numpy as np
import nltk
from typing import List, Tuple, Callable
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


def dynamic_threshold_iqr(data: List[float]) -> float:
    """
    Calculate dynamic threshold using Interquartile Range (IQR) method.
    
    Args:
        data: List of numerical values
        
    Returns:
        Upper threshold value (Q3 + 1.5 * IQR)
    """
    if len(data) == 0:
        return 0.0
    
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    upper_threshold = q3 + 1.5 * iqr
    return upper_threshold


def detect_rules_and_roleplay(prompt: str) -> bool:
    """
    Identify if a prompt contains both rules and roleplay instructions.
    
    This combination can indicate manipulation attempts where adversaries
    try to bypass safety guidelines through creative roleplay scenarios.
    
    Args:
        prompt: Input text to analyze
        
    Returns:
        True if prompt contains both rules and roleplay keywords
        
    Example:
        >>> prompt = "Imagine you are a hacker, but you must follow these rules..."
        >>> detect_rules_and_roleplay(prompt)
        True
    """
    rules_keywords = ['must', 'must not', 'should', 'should not', 'require', 'mandatory']
    roleplay_keywords = ['imagine', 'pretend', 'roleplay', 'act as', 'you are']
    
    prompt_lower = prompt.lower()
    contains_rules = any(re.search(r'\b' + keyword.lower() + r'\b', prompt_lower) 
                        for keyword in rules_keywords)
    contains_roleplay = any(re.search(r'\b' + keyword.lower() + r'\b', prompt_lower) 
                           for keyword in roleplay_keywords)
    
    return contains_rules and contains_roleplay


def detect_color_change_code(text: str) -> Tuple[bool, List[str]]:
    """
    Identify HTML or CSS color-changing code in text.
    
    Color-changing code can be used to hide malicious instructions or
    create visual distractions in prompt injection attacks.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Tuple of (contains_code: bool, detected_colors: List[str])
        
    Example:
        >>> text = '<span style="color: red;">Hidden text</span>'
        >>> contains, colors = detect_color_change_code(text)
        >>> contains
        True
    """
    html_pattern = r'<span\s+style="color:\s*([^"]+)">'
    css_pattern = r'\bcolor:\s*([^;]+)\b'
    
    matches = []
    for pattern in [html_pattern, css_pattern]:
        matches.extend(re.findall(pattern, text))
    
    contains_color_change_code = len(matches) > 0
    return contains_color_change_code, matches


def detect_nlp_outliers(paragraph: str, 
                       dynamic_threshold: Callable[[List[float]], float],
                       model: SentenceTransformer = None) -> List[str]:
    """
    Identify semantically anomalous sentences using sentence embeddings.
    
    Computes embeddings for each sentence and finds those with low
    average cosine similarity to other sentences in the paragraph.
    
    Args:
        paragraph: Input text containing multiple sentences
        dynamic_threshold: Function to calculate outlier threshold
        model: Pre-loaded SentenceTransformer model (optional)
        
    Returns:
        List of outlier sentences
        
    Example:
        >>> text = "Hello world. How are you? XYZABC malicious code inject."
        >>> outliers = detect_nlp_outliers(text, dynamic_threshold_iqr)
    """
    sentences = nltk.sent_tokenize(paragraph)
    
    if len(sentences) <= 1:
        return []
    
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    embeddings = model.encode(sentences)
    similarities = cosine_similarity(embeddings)
    mean_similarities = np.mean(similarities, axis=1)
    
    threshold = dynamic_threshold(mean_similarities)
    outliers = [sentences[i] for i, sim in enumerate(mean_similarities) if sim < threshold]
    
    return outliers


def detect_length_outliers(text: str,
                          dynamic_threshold: Callable[[List[int]], float]) -> List[str]:
    """
    Identify sentences with anomalous lengths.
    
    Unusually long sentences can contain hidden instructions or
    attempts to overwhelm context windows.
    
    Args:
        text: Input text to analyze
        dynamic_threshold: Function to calculate outlier threshold
        
    Returns:
        List of outlier sentences
    """
    sentences = nltk.sent_tokenize(text)
    lengths = [len(sentence) for sentence in sentences]
    
    if len(lengths) == 0:
        return []
    
    threshold = dynamic_threshold(lengths)
    outliers = [sentences[i] for i, length in enumerate(lengths) if length > threshold]
    
    return outliers


def detect_word_frequency_outliers(text: str,
                                  dynamic_threshold: Callable[[List[int]], float]) -> List[str]:
    """
    Identify words with anomalously high frequencies.
    
    Repeated words can indicate spam, injection attempts, or
    adversarial patterns designed to bias model behavior.
    
    Args:
        text: Input text to analyze
        dynamic_threshold: Function to calculate outlier threshold
        
    Returns:
        List of outlier words
    """
    words = nltk.word_tokenize(text)
    freq_dist = nltk.FreqDist(words)
    
    if len(freq_dist) == 0:
        return []
    
    threshold = dynamic_threshold(list(freq_dist.values()))
    outliers = [word for word, freq in freq_dist.items() if freq > threshold]
    
    return outliers


def detect_statistical_outliers(data: List[float],
                               dynamic_threshold: Callable[[List[float]], float]) -> List[float]:
    """
    Identify statistical outliers in numerical data.
    
    Args:
        data: List of numerical values
        dynamic_threshold: Function to calculate outlier threshold
        
    Returns:
        List of outlier values
    """
    if len(data) == 0:
        return []
    
    threshold = dynamic_threshold(data)
    outliers = [x for x in data if x > threshold]
    
    return outliers


def detect_all_text_anomalies(prompt: str,
                              model: SentenceTransformer = None) -> dict:
    """
    Run all text-based anomaly detection methods on a prompt.
    
    Args:
        prompt: Input text to analyze
        model: Pre-loaded SentenceTransformer model (optional)
        
    Returns:
        Dictionary with detection results for each method
    """
    results = {
        'rules_and_roleplay': detect_rules_and_roleplay(prompt),
        'color_change_code': detect_color_change_code(prompt)[0],
        'color_matches': detect_color_change_code(prompt)[1],
        'nlp_outliers': detect_nlp_outliers(prompt, dynamic_threshold_iqr, model),
        'length_outliers': detect_length_outliers(prompt, dynamic_threshold_iqr),
        'word_frequency_outliers': detect_word_frequency_outliers(prompt, dynamic_threshold_iqr),
    }
    
    # Calculate anomaly score (number of triggered detectors)
    anomaly_flags = [
        results['rules_and_roleplay'],
        results['color_change_code'],
        len(results['nlp_outliers']) > 0,
        len(results['length_outliers']) > 0,
        len(results['word_frequency_outliers']) > 0
    ]
    results['anomaly_score'] = sum(anomaly_flags)
    results['is_anomalous'] = results['anomaly_score'] >= 2  # Threshold: 2+ detectors
    
    return results
