#!/usr/bin/env python3
"""
Test script for flexible product code matching in prompt_static_analyzer.py
Tests various input formats for product codes to ensure they are properly detected.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

from agents.prompt_static_analyzer.prompt_static_analyzer import ProductCodeDetector, QueryRouter, PromptStaticAnalyzer

def test_product_code_variations():
    """Test different variations of product code formatting"""
    print("="*80)
    print("TESTING PRODUCT CODE VARIATIONS")
    print("="*80)
    
    detector = ProductCodeDetector()
    
    # Test cases for various formatting variations
    test_cases = [
        # Equipment codes
        ("MS021", "Standard equipment code"),
        ("ms021", "Lowercase equipment code"),
        ("MS 021", "Equipment code with space"),
        ("ms 021", "Lowercase equipment with space"),
        
        # Equipment codes with suffixes
        ("MS005A", "Equipment code with letter suffix"),
        ("ms005a", "Lowercase equipment with letter suffix"),
        ("MS 005A", "Equipment with space and letter suffix"),
        
        # Equipment codes with COM
        ("MS004COM", "Equipment with COM attached"),
        ("ms004com", "Lowercase equipment with COM"),
        ("MS004 COM", "Equipment with spaced COM"),
        ("ms004 com", "Lowercase equipment with spaced COM"),
        ("MS 004 COM", "Equipment with multiple spaces and COM"),
        
        # Tool codes
        ("MS50039-HPS", "Standard tool code"),
        ("ms50039-hps", "Lowercase tool code"),
        ("MS50039HPS", "Tool code without hyphen"),
        ("ms50039hps", "Lowercase tool without hyphen"),
        ("MS50039 HPS", "Tool code with space instead of hyphen"),
        ("ms50039 hps", "Lowercase tool with space"),
        ("MS 50039 - HPS", "Tool code with multiple spaces"),
        ("ms 50039 - hps", "Lowercase tool with multiple spaces"),
        
        # KIT codes
        ("KIT005A", "Standard KIT code"),
        ("kit005a", "Lowercase KIT code"),
        ("KIT 005A", "KIT code with space"),
        ("kit 005a", "Lowercase KIT with space"),
        
        # Special cases
        ("LOKI", "Special LOKI code"),
        ("loki", "Lowercase LOKI"),
        
        # Mixed queries
        ("I need MS 021 and ms50039 hps for testing", "Mixed case multiple codes"),
        ("Can MS004COM work with ms 50039-HPS?", "Mixed formats in question"),
    ]
    
    for query, description in test_cases:
        print(f"\nTesting: {description}")
        print(f"Input: '{query}'")
        result = detector.detect_product_codes(query)
        print(f"Equipment: {result['equipment']}")
        print(f"Tools: {result['tools']}")
        print(f"Unknown: {result['unknown']}")
        print("-" * 40)

def test_normalization_function():
    """Test the normalization function directly"""
    print("\n" + "="*80)
    print("TESTING NORMALIZATION FUNCTION")
    print("="*80)
    
    detector = ProductCodeDetector()
    
    normalization_cases = [
        "MS 021",
        "ms004 com",
        "MS50039 HPS",
        "ms50039hps",
        "MS50039-HPS",
        "ms 50039 - hps",
        "KIT 005A",
        "kit005a"
    ]
    
    for case in normalization_cases:
        normalized = detector._normalize_product_code(case)
        print(f"'{case}' -> '{normalized}'")

def test_full_query_routing():
    """Test complete query routing with flexible codes"""
    print("\n" + "="*80)
    print("TESTING FULL QUERY ROUTING")
    print("="*80)
    
    analyzer = PromptStaticAnalyzer()
    
    test_queries = [
        "Can I use ms 021 for testing generators?",
        "What about MS004COM equipment specifications?", 
        "MS50039 HPS tool availability and pricing?",
        "Difference between ms50039hps and MS50039-HPS?",
        "Need cable for MS 112 and ms 800a equipment",
        "How to test with ms 005a and MS50039 HPS together?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = analyzer.route_query(query)
        print(f"Specialists: {', '.join(result['specialists'])}")
        print(f"Confidence: {result['final_confidence']:.2f}")
        
        analysis = result['analysis']
        if analysis['product_codes']['equipment']:
            print(f"Equipment codes: {', '.join(analysis['product_codes']['equipment'])}")
        if analysis['product_codes']['tools']:
            print(f"Tool codes: {', '.join(analysis['product_codes']['tools'])}")
        print("-" * 60)

def test_edge_cases():
    """Test edge cases and potential issues"""
    print("\n" + "="*80)
    print("TESTING EDGE CASES")
    print("="*80)
    
    detector = ProductCodeDetector()
    
    edge_cases = [
        ("", "Empty string"),
        ("MS", "Just MS prefix"),
        ("MS123456", "Invalid format - too many digits"),
        ("XS021", "Wrong prefix"),
        ("MS021 MS050", "Multiple equipment codes"),
        ("MS50039HPS MS51001HPS", "Multiple tool codes"),
        ("This MS021 and that MS50039-HPS", "Codes in sentence"),
        ("MS021,MS800A,MS50039-HPS", "Comma separated codes"),
    ]
    
    for query, description in edge_cases:
        print(f"\nTesting: {description}")
        print(f"Input: '{query}'")
        try:
            result = detector.detect_product_codes(query)
            print(f"Equipment: {result['equipment']}")
            print(f"Tools: {result['tools']}")
            print(f"Unknown: {result['unknown']}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 40)

def main():
    """Run all tests"""
    print("FLEXIBLE PRODUCT CODE MATCHING TEST SUITE")
    print("Testing prompt_static_analyzer.py flexible matching capabilities")
    
    try:
        test_normalization_function()
        test_product_code_variations()
        test_full_query_routing()
        test_edge_cases()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED")
        print("="*80)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()