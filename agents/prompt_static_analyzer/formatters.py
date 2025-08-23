def format_analyzer_output_for_orchestrator(analyzer_output):
    """
    Reformulates static prompt analyzer output into a structured message 
    for the Orchestrator Assistant to process routing decisions.
    
    Args:
        analyzer_output (dict): Dictionary containing analyzer results with keys:
            - specialists: list of recommended specialists
            - analysis: detailed analysis data
            - keyword_routing: keyword-based routing info
            - final_confidence: overall confidence score
            - routing_context: formatted context string
    
    Returns:
        str: Formatted message for the Orchestrator Assistant
    """
    
    # Extract main components
    specialists = analyzer_output.get('specialists', [])
    analysis = analyzer_output.get('analysis', {})
    keyword_routing = analyzer_output.get('keyword_routing', {})
    final_confidence = analyzer_output.get('final_confidence', 0.0)
    routing_context = analyzer_output.get('routing_context', '')
    
    # Build the message
    message_parts = []
    
    # Header with pre-analysis summary
    message_parts.append("=== PRE-ANALYSIS ROUTING INTELLIGENCE ===")
    message_parts.append(f"Static Analyzer Recommendations: {', '.join(specialists)}")
    message_parts.append(f"Overall Confidence: {final_confidence:.2f}")
    message_parts.append("")
    
    # Detailed analysis breakdown
    if analysis:
        message_parts.append("=== DETAILED ANALYSIS ===")
        
        # Primary domains detected
        primary_domains = analysis.get('primary_domains', [])
        if primary_domains:
            message_parts.append(f"Primary Domains Detected: {', '.join(primary_domains)}")
        
        # Product codes
        product_codes = analysis.get('product_codes', {})
        if product_codes:
            message_parts.append("Product Codes Identified:")
            for category, codes in product_codes.items():
                if codes:  # Only show non-empty categories
                    message_parts.append(f"  {category}: {', '.join(codes)}")
        
        # OEM numbers
        oem_numbers = analysis.get('oem_numbers', [])
        if oem_numbers:
            message_parts.append(f"OEM Numbers Detected: {', '.join(oem_numbers)}")
        
        # Car brands
        car_brands = analysis.get('car_brands', [])
        if car_brands:
            message_parts.append(f"Vehicle Brands Mentioned: {', '.join(car_brands)}")
        
        # Routing confidence from analysis
        routing_confidence = analysis.get('routing_confidence', 0.0)
        message_parts.append(f"Analysis Routing Confidence: {routing_confidence:.2f}")
        message_parts.append("")
    
    # Keyword routing details
    if keyword_routing:
        message_parts.append("=== KEYWORD ROUTING ANALYSIS ===")
        
        keyword_specialists = keyword_routing.get('specialists', [])
        if keyword_specialists:
            message_parts.append(f"Keyword-based Specialists: {', '.join(keyword_specialists)}")
        
        reasoning = keyword_routing.get('reasoning', '')
        if reasoning:
            message_parts.append(f"Keyword Reasoning: {reasoning}")
        
        keyword_confidence = keyword_routing.get('confidence', 0.0)
        message_parts.append(f"Keyword Confidence: {keyword_confidence:.2f}")
        message_parts.append("")
    
    # Final instructions for orchestrator
    message_parts.append("=== ORCHESTRATOR INSTRUCTIONS ===")
    message_parts.append("Please consider this pre-analysis when making your routing decision:")
    message_parts.append("1. Review the static analyzer's specialist recommendations above")
    message_parts.append("2. Consider the confidence scores and detected entities")
    message_parts.append("3. Validate against your own analysis of the user query")
    message_parts.append("4. Make final routing decision based on combined intelligence")
    
    if final_confidence < 0.5:
        message_parts.append("⚠️  LOW CONFIDENCE WARNING: Static analysis confidence is below 0.5")
        message_parts.append("   Consider routing to 'support' for clarification if query remains unclear")
    elif final_confidence > 0.8:
        message_parts.append("✅ HIGH CONFIDENCE: Static analysis shows strong routing indicators")
    
    return "\n".join(message_parts)