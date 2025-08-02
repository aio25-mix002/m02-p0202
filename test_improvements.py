#!/usr/bin/env python3
"""
Simple test script to validate text processing pipeline improvements.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import utils


def test_improved_stemming():
    """Test improved stemming with different algorithms."""
    print("Testing improved stemming...")
    
    # Use words where Porter and Snowball differ
    tokens = ["generational", "relational", "conditional", "traditional"]
    
    # Test Porter stemmer (default)
    porter_results = utils.stemming(tokens, method="porter")
    print(f"Porter stemming: {porter_results}")
    
    # Test Snowball stemmer
    snowball_results = utils.stemming(tokens, method="snowball")
    print(f"Snowball stemming: {snowball_results}")
    
    # Test basic functionality works
    assert isinstance(porter_results, list), "Porter stemming should return a list"
    assert isinstance(snowball_results, list), "Snowball stemming should return a list"
    assert len(porter_results) == len(tokens), "Should return same number of tokens"
    assert len(snowball_results) == len(tokens), "Should return same number of tokens"
    
    print("✓ Improved stemming test passed")


def test_lemmatization():
    """Test lemmatization functionality."""
    print("\nTesting lemmatization...")
    
    tokens = ["running", "flies", "dogs", "houses", "better"]
    lemmatized = utils.lemmatization(tokens)
    print(f"Lemmatized tokens: {lemmatized}")
    
    # Basic checks
    assert isinstance(lemmatized, list), "Lemmatization should return a list"
    assert len(lemmatized) == len(tokens), "Should return same number of tokens"
    print("✓ Lemmatization test passed")


def test_feature_extraction():
    """Test feature extraction functionality."""
    print("\nTesting feature extraction...")
    
    # Test text with various features
    test_text = "URGENT! Visit http://spam.com or call 555-123-4567. Email contact@spam.com for $1000 prize!"
    features = utils.extract_text_features(test_text)
    
    print(f"Extracted features: {features}")
    
    # Verify expected features
    assert features['has_url'] == 1, "Should detect URL"
    assert features['has_phone'] == 1, "Should detect phone number"
    assert features['has_email'] == 1, "Should detect email"
    assert features['has_currency'] == 1, "Should detect currency symbol"
    assert features['capital_ratio'] > 0, "Should detect capital letters"
    assert features['exclamation_count'] > 0, "Should count exclamation marks"
    
    print("✓ Feature extraction test passed")


def test_individual_detectors():
    """Test individual detector functions."""
    print("\nTesting individual detector functions...")
    
    # Test URL detection
    assert utils.has_url("Visit http://example.com"), "Should detect HTTP URL"
    assert utils.has_url("Check https://secure.site.org"), "Should detect HTTPS URL"
    assert not utils.has_url("No URL here"), "Should not detect URL when none present"
    
    # Test phone detection
    assert utils.has_phone("Call 555-123-4567"), "Should detect phone number"
    assert utils.has_phone("Phone: (555) 123-4567"), "Should detect formatted phone"
    assert not utils.has_phone("No phone here"), "Should not detect phone when none present"
    
    # Test email detection
    assert utils.has_email("Contact test@example.com"), "Should detect email"
    assert not utils.has_email("No email here"), "Should not detect email when none present"
    
    print("✓ Individual detector tests passed")


def test_preprocessing_options():
    """Test preprocessing with new options."""
    print("\nTesting preprocessing with new options...")
    
    text = "The running dogs are flying quickly!"
    
    # Test default (stemming with porter)
    default_result = utils.preprocess_text(text)
    print(f"Default preprocessing: {default_result}")
    
    # Test with snowball stemming
    snowball_result = utils.preprocess_text(text, semantic_method="stemming", stemming_method="snowball")
    print(f"Snowball stemming: {snowball_result}")
    
    # Test with lemmatization
    lemma_result = utils.preprocess_text(text, semantic_method="lemmatization")
    print(f"Lemmatization: {lemma_result}")
    
    # Verify they're different
    assert default_result != lemma_result, "Stemming and lemmatization should produce different results"
    print("✓ Preprocessing options test passed")


def test_backward_compatibility():
    """Test that old code still works."""
    print("\nTesting backward compatibility...")
    
    text = "This is a test message."
    
    # This should work exactly as before
    old_style_result = utils.preprocess_text(text)
    
    # This should produce the same result
    new_style_result = utils.preprocess_text(text, semantic_method="stemming", stemming_method="porter")
    
    assert old_style_result == new_style_result, "Backward compatibility broken"
    print("✓ Backward compatibility test passed")


def main():
    """Run all tests."""
    print("Running text processing pipeline improvement tests...\n")
    
    try:
        test_improved_stemming()
        test_lemmatization()
        test_feature_extraction()
        test_individual_detectors()
        test_preprocessing_options()
        test_backward_compatibility()
        
        print("\n🎉 All tests passed! Text processing pipeline improvements are working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()