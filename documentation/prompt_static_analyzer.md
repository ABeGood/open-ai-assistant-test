# Prompt Static Analyzer

## Overview

The `prompt_static_analyzer.py` module provides a comprehensive static analysis system for automotive diagnostic equipment customer support queries. It combines pattern matching, keyword detection, and product code recognition to intelligently route customer queries to appropriate specialist assistants.

## Architecture

The module consists of three main classes that work together to provide intelligent query routing:

### 1. ProductCodeDetector
Handles detection and categorization of product codes and keywords in customer queries.

### 2. QueryRouter  
Analyzes queries comprehensively using the ProductCodeDetector and provides routing decisions.

### 3. PromptStaticAnalyzer
Main orchestrator that combines static analysis with LLM-enhanced routing decisions.

## Class Documentation

### ProductCodeDetector

The `ProductCodeDetector` class is responsible for identifying various elements in customer queries related to automotive diagnostic equipment.

#### Key Features:

**Product Code Detection:**
- **Equipment Codes**: Detects MS-series equipment (MS021, MS005A, MS800, etc.)
- **Tool Codes**: Identifies specialized tool codes with HPS/EPS suffixes (MS51001-HPS, MS50090-EPS, etc.)
- **Kit Codes**: Recognizes kit identifiers (KIT005A, etc.)

**Multi-language Keyword Detection:**
- Supports Russian, Ukrainian, and English
- Domain-specific keyword categories:
  - `equipment`: Equipment specifications and comparisons
  - `diagnostics`: Testing procedures and diagnostic methods
  - `compatibility`: Vehicle compatibility and OEM parts
  - `cables`: Cable connections and wiring
  - `tools`: Hand tools and accessories
  - `support`: General support and troubleshooting

**Pattern Recognition:**
- **OEM Numbers**: Detects automotive part numbers (5190161AK, 1501256-00-G)
- **Car Brands**: Identifies vehicle manufacturers (BMW, Mercedes, Ford, etc.)

#### Methods:

```python
def detect_product_codes(self, query: str) -> Dict[str, List[str]]
```
Detects and categorizes product codes in the query using regex patterns.

```python  
def detect_domain_keywords(self, query: str) -> Dict[str, int]
```
Counts keyword matches for each domain across all supported languages.

```python
def detect_oem_numbers(self, query: str) -> List[str]
```
Identifies OEM part numbers using specific patterns.

```python
def detect_car_brands(self, query: str) -> List[str] 
```
Detects mentions of automotive brands in the query.

### QueryRouter

The `QueryRouter` class provides comprehensive query analysis by orchestrating the `ProductCodeDetector`.

#### Methods:

```python
def analyze_query(self, query: str) -> Dict
```
Performs complete query analysis and returns structured results including:
- Detected product codes (equipment/tools/unknown)
- Domain keyword scores
- OEM numbers and car brands
- Primary routing domains
- Confidence score

```python
def _calculate_confidence(self, domain_scores: Dict, product_codes: Dict) -> float
```
Calculates routing confidence based on detected codes and keyword matches:
- Product codes detected: 0.5-0.9 confidence
- Multiple keywords: 0.4-0.8 confidence  
- Single keyword: 0.6 confidence
- No matches: 0.3 confidence

### PromptStaticAnalyzer

The main orchestration class that combines static analysis with enhanced context creation for LLM routing.

#### Methods:

```python
def route_query(self, user_query: str) -> Dict
```
Main routing function that:
1. Analyzes query with Python detection
2. Creates enhanced context for LLM
3. Gets keyword-based routing decision
4. Combines analysis results

```python
def _create_enhanced_context(self, query: str, analysis: Dict) -> str
```
Creates detailed context string including all detected information for LLM processing.

```python
def _combine_routing_decisions(self, analysis: Dict, keyword_routing: Dict) -> Dict
```
Merges static analysis results with keyword-based routing to produce final specialist assignments.

## Usage Examples

### Basic Query Analysis

```python
from prompt_static_analyzer import PromptStaticAnalyzer

analyzer = PromptStaticAnalyzer()

# Analyze a customer query
query = "Какой стенд для генераторов ты посоветуешь для маленькой мастерской?"
result = analyzer.route_query(query)

print(f"Specialists: {result['specialists']}")
print(f"Confidence: {result['final_confidence']}")
```

### Product Code Detection

```python
from prompt_static_analyzer import ProductCodeDetector

detector = ProductCodeDetector()
codes = detector.detect_product_codes("MS021 и MS800A equipment")

print(codes)
# Output: {'equipment': ['MS021', 'MS800A'], 'tools': [], 'unknown': []}
```

## Supported Product Codes

### Equipment Codes
- MS-series main equipment: MS021, MS005A, MS015A, MS800, MS900, etc.
- Communication variants: MS002 COM, MS004 COM, MS012 COM, etc.
- Special equipment: LOKI, MS561 PRO

### Tool Codes  
- HPS (Hydraulic Power Steering) tools: MS51001-HPS, MS50006-HPS, etc.
- EPS (Electric Power Steering) tools: MS50009-EPS, MS50175-EPS, etc.
- Various specialized tools with MS5xxxx-XXX format

## Routing Logic

The system uses a multi-layered approach to determine appropriate specialists:

1. **Product Code Analysis**: Direct mapping of detected codes to specialist domains
2. **Keyword Scoring**: Multi-language keyword matching with domain scoring
3. **Special Rules**: Custom logic for compatibility queries (OEM numbers, car brands)
4. **Fallback**: Routes to 'support' domain when no clear match is found

### Domain Mapping

- **equipment**: Equipment specifications, comparisons, recommendations
- **diagnostics**: Testing procedures, parameter measurement, troubleshooting  
- **compatibility**: Vehicle compatibility, OEM part verification
- **cables**: Cable connections, wiring, pinouts
- **tools**: Hand tools, accessories, specialized instruments
- **support**: General help, warranty, training, non-working equipment

## Multi-language Support

The system supports customer queries in:
- **Russian (ru)**: Primary language with comprehensive keyword coverage
- **Ukrainian (uk)**: Full keyword support with regional variations
- **English (en)**: International support with technical terminology

## Integration

This module integrates with the larger customer support system by:

1. **Pre-processing queries** before LLM analysis
2. **Providing enhanced context** to specialist assistants  
3. **Confidence scoring** for routing decisions
4. **Structured output** for downstream processing

## Error Handling

The system is designed to be robust:
- Unknown product codes are categorized separately
- Fallback routing ensures all queries receive responses  
- Multi-language support handles mixed-language queries
- Confidence scoring helps identify uncertain routing decisions

## Performance Considerations

- **Regex-based detection** provides fast pattern matching
- **Static keyword lists** enable efficient domain scoring
- **Caching opportunities** exist for repeated query patterns
- **Minimal dependencies** on external services for core functionality

## Testing

The module includes a comprehensive test suite with real customer query examples:

```python
test_queries = [
    "Какой стенд для генераторов ты посоветуешь для маленькой мастерской?",
    "как перевірити діодний міст тестером MS021?", 
    "Генератор от Jeep Wrangler 5190161AK можно проверить на MS005A?",
    # ... more examples
]
```

Each test demonstrates different aspects of the routing logic including equipment recommendations, diagnostic procedures, compatibility checks, and support requests.