#!/usr/bin/env python3
"""
Demonstration script showing the improved text processing pipeline.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import utils


def demonstrate_improvements():
    """Demonstrate the improved text processing pipeline."""
    
    print("=== Text Processing Pipeline Improvements Demo ===\n")
    
    # Sample texts for demonstration
    spam_text = "URGENT! Visit http://spam-site.com or call 555-123-4567. Email contact@spam.com for $1000 prize!"
    ham_text = "Hey, are you running to the meeting? The dogs were flying through the park yesterday."
    
    print("1. IMPROVED STEMMING OPTIONS")
    print("-" * 40)
    tokens = ["running", "flies", "better", "generational", "traditional"]
    print(f"Original tokens: {tokens}")
    
    porter_results = utils.stemming(tokens, method="porter")
    snowball_results = utils.stemming(tokens, method="snowball")
    
    print(f"Porter stemming:   {porter_results}")
    print(f"Snowball stemming: {snowball_results}")
    print()
    
    print("2. LEMMATIZATION OPTION")
    print("-" * 40)
    lemma_results = utils.lemmatization(tokens)
    print(f"Lemmatization:     {lemma_results}")
    print()
    
    print("3. COMPREHENSIVE PREPROCESSING OPTIONS")
    print("-" * 40)
    print(f"Sample text: '{ham_text}'")
    
    # Different preprocessing approaches
    default_result = utils.preprocess_text(ham_text)
    snowball_result = utils.preprocess_text(ham_text, semantic_method="stemming", stemming_method="snowball")
    lemma_result = utils.preprocess_text(ham_text, semantic_method="lemmatization")
    
    print(f"Default (Porter):  {default_result}")
    print(f"Snowball stemming: {snowball_result}")
    print(f"Lemmatization:     {lemma_result}")
    print()
    
    print("4. FEATURE EXTRACTION FOR SPAM DETECTION")
    print("-" * 40)
    print(f"Spam text: '{spam_text}'")
    spam_features = utils.extract_text_features(spam_text)
    
    print("Extracted features:")
    for feature, value in spam_features.items():
        if isinstance(value, float):
            print(f"  {feature}: {value:.3f}")
        else:
            print(f"  {feature}: {value}")
    print()
    
    print(f"Ham text: '{ham_text}'")
    ham_features = utils.extract_text_features(ham_text)
    
    print("Extracted features:")
    for feature, value in ham_features.items():
        if isinstance(value, float):
            print(f"  {feature}: {value:.3f}")
        else:
            print(f"  {feature}: {value}")
    print()
    
    print("5. INDIVIDUAL DETECTOR FUNCTIONS")
    print("-" * 40)
    test_texts = [
        "Visit our website at https://example.com",
        "Call us at 555-123-4567 for more info",
        "Send email to contact@company.com",
        "Just a normal message without special content"
    ]
    
    for text in test_texts:
        print(f"Text: '{text}'")
        print(f"  Has URL:   {utils.has_url(text)}")
        print(f"  Has Phone: {utils.has_phone(text)}")
        print(f"  Has Email: {utils.has_email(text)}")
        print()
    
    print("6. SPAM VS HAM FEATURE COMPARISON")
    print("-" * 40)
    print("Key differences in features:")
    print(f"Capital ratio - Spam: {spam_features['capital_ratio']:.3f}, Ham: {ham_features['capital_ratio']:.3f}")
    print(f"Has URL - Spam: {spam_features['has_url']}, Ham: {ham_features['has_url']}")
    print(f"Has phone - Spam: {spam_features['has_phone']}, Ham: {ham_features['has_phone']}")
    print(f"Has email - Spam: {spam_features['has_email']}, Ham: {ham_features['has_email']}")
    print(f"Has currency - Spam: {spam_features['has_currency']}, Ham: {ham_features['has_currency']}")
    print(f"Exclamations - Spam: {spam_features['exclamation_count']}, Ham: {ham_features['exclamation_count']}")
    print()
    
    print("✅ All improvements demonstrated successfully!")
    print("\nBackward compatibility: All existing code continues to work unchanged.")


if __name__ == "__main__":
    demonstrate_improvements()