import re
import json
from typing import List, Dict, Set, Tuple

class ProductCodeDetector:
    def __init__(self):
        # Equipment product codes
        self.equipment_codes = {
            'MS021', 'MS005A', 'MS015A', 'KIT005A', 'MS008', 'MS002A', 'MS002 COM', 
            'MS004 COM', 'MS005', 'MS006', 'MS012 COM', 'MS013 COM', 'MS016', 'MS016A', 
            'MS015', 'MS014', 'MS111', 'MS112', 'MS121', 'MS101P', 'MS201', 'MS203', 
            'MS200', 'MS301', 'MS300', 'MS561 PRO', 'MS550', 'MS570', 'MS521', 'MS510', 
            'MS511', 'MS504', 'MS502M', 'MS522', 'MS503N', 'MS800', 'MS800A', 'MS801', 
            'MS900', 'LOKI', 'MS402', 'MS410'
        }
        
        # Tool product codes
        self.tool_codes = {
            'MS51001-HPS', 'MS51002-HPS', 'MS51003-HPS', 'MS51004-HPS', 'MS51005-HPS',
            'MS50006-HPS', 'MS50013-HPS', 'MS50007-HPS', 'MS50008-HPS', 'MS50147-HPS',
            'MS50009-EPS', 'MS50090-EPS', 'MS50175-EPS', 'MS50181-EPS', 'MS50010-HPS',
            'MS50035-HPS', 'MS50037-HPS', 'MS50043-HPS', 'MS50089-HPS', 'MS50133-HPS',
            'MS50136-HPS', 'MS50011-EPS', 'MS50029-EPS', 'MS50034-EPS', 'MS50036-EPS',
            'MS50079-EPS', 'MS50084-EPS', 'MS50116-EPS', 'MS50146-EPS', 'MS50012-HPS',
            'MS50021-HPS', 'MS50023-HPS', 'MS50024-HPS', 'MS50027-HPS', 'MS50028-HPS',
            'MS50038-HPS', 'MS50039-HPS', 'MS50044-HPS', 'MS50064-HPS', 'MS50065-HPS',
            'MS50066-HPS', 'MS50067-HPS', 'MS50068-HPS', 'MS51014-HPS', 'MS51015-HPS',
            'MS51016-HPS', 'MS51017-HPS', 'MS50018-HPS', 'MS50019-HPS', 'MS50025-HPS',
            'MS50033-HPS', 'MS50020-EPS', 'MS50022-EPS', 'MS50026-EPS', 'MS50031-EPS',
            'MS50072-EPS', 'MS50092-EPS', 'MS50094-EPS', 'MS50098-EPS', 'MS50102-EPS',
            'MS50103-EPS', 'MS50137-EPS', 'MS50165-EPS', 'MS50174-EPS', 'MS50177-EPS',
            'MS50178-EPS', 'MS50023-EPS', 'MS50097-EPS', 'MS50153-EPS', 'MS5230-HPS',
            'MS50032-HPS', 'MS51040-HPS', 'MS50041-EPS', 'MS51042-HPS', 'MS53045-HPS',
            'MS53060-HPS', 'MS53061-HPS', 'MS53111-HPS', 'MS53128-HPS', 'MS52046-HPS',
            'MS50047-HPS', 'MS50048-HPS', 'MS53050-HPS', 'MS53051-HPS', 'MS53052-HPS',
            'MS53053-HPS', 'MS53054-HPS', 'MS52055-HPS', 'MS52056-HPS', 'MS52057-HPS',
            'MS51058-HPS', 'MS51059-HPS', 'MS53049-HPS', 'MS53062-HPS', 'MS53063-HPS',
            'MS53069-HPS', 'MS53070-HPS', 'MS53114-HPS', 'MS53127-HPS', 'MS53130-HPS',
            'MS53140-HPS', 'MS50071-HPS', 'MS50073-HPS', 'MS50077-HPS', 'MS50078-HPS',
            'MS50083-HPS', 'MS50091-HPS', 'MS50096-HPS', 'MS50101-HPS', 'MS50104-HPS',
            'MS50105-HPS', 'MS50107-HPS', 'MS50150-HPS', 'MS52074-HPS', 'MS52075-HPS',
            'MS52076-HPS', 'MS51080-HPS', 'MS51081-HPS', 'MS51082-HPS', 'MS50085-HPS',
            'MS50086-HPS', 'MS50087-HPS', 'MS50088-HPS', 'MS52093-HPS', 'MS50095-EPS',
            'MS5299-EPS', 'MS51100-HPS', 'MS52106-HPS', 'MS52108-EPS', 'MS52109-HPS',
            'MS5210-HPS', 'MS5312-HPS', 'MS52113-HPS', 'MS52115-HPS', 'MS50117-EPS',
            'MS52118-HPS', 'MS52119-HPS', 'MS52120-HPS', 'MS52121-HPS', 'MS52122-HPS',
            'MS52123-HPS', 'MS52124-HPS', 'MS52125-HPS', 'MS52126-HPS', 'MS52129-HPS',
            'MS50131-HPS', 'MS5203-HPS', 'MS50134-EPS', 'MS50135-EPS', 'MS50138-EPS',
            'MS50139-EPS', 'MS52141-EPS', 'MS51142-EPS', 'MS51143-EPS', 'MS50144-EPS',
            'MS50145-EPS', 'MS50148-EPS', 'MS50149-EPS', 'MS50173-EPS', 'MS50179-EPS',
            'MS50180-EPS', 'MS50182-EPS', 'MS50188-EPS', 'MS50151-EPS', 'MS50152-EPS',
            'MS50156-EPS', 'MS50154-EPS', 'MS50155-EPS', 'MS50157-EPS', 'MS50158-EPS',
            'MS50159-EPS', 'MS50160-EPS', 'MS50162-EPS', 'MS50161-EPS', 'MS50163-EPS',
            'MS50164-EPS', 'MS50166-EPS', 'MS50167-EPS', 'MS50169-EPS', 'MS50168-EPS',
            'MS50170-EPS', 'MS50171-EPS', 'MS50172-EPS', 'MS50176-EPS', 'MS50183-EPS',
            'MS50184-EPS', 'MS51185-EPS', 'MS51185A-EPS', 'MS54185A-EPS', 'MS50186-EPS',
            'MS50187-EPS', 'MS50189-EPS', 'MS50190-EPS', 'MS50191-EPS', 'MS50192-EPS',
            'MS5353-HPS', 'MS5354-HPS', 'MS5355-HPS', 'MS5001-HPS', 'MS5202-HPS'
        }
        
        # Routing keywords by domain
        self.domain_keywords = {
            'equipment': {
                'ru': ['стенд', 'оборудование', 'модель', 'нагрузка', 'давление', 'максимальная', 'спецификации', 'отличается'],
                'uk': ['стенд', 'обладнання', 'модель', 'навантаження', 'тиск', 'максимальний', 'специфікації', 'відрізняється'],
                'en': ['stand', 'equipment', 'model', 'load', 'pressure', 'maximum', 'specifications', 'compare', 'difference', 'recommend']
            },
            'diagnostics': {
                'ru': ['проверить', 'тест', 'измерить', 'параметры', 'диагностика', 'режим', 'ручной', 'автоматический', 'выпрямляч'],
                'uk': ['перевірити', 'тест', 'виміряти', 'параметри', 'діагностика', 'режим', 'ручний', 'автоматичний', 'випрямляч', 'діодний'],
                'en': ['test', 'check', 'measure', 'parameters', 'diagnostic', 'mode', 'manual', 'automatic', 'rectifier', 'diode']
            },
            'compatibility': {
                'ru': ['генератор от', 'компрессор с', 'марки автомобилей', 'OEM', 'модели автомобилей', 'можно проверить'],
                'uk': ['генератор від', 'компресор з', 'марки автомобілів', 'OEM', 'моделі автомобілів', 'можна перевірити'],
                'en': ['generator from', 'compressor from', 'car brands', 'vehicle models', 'OEM', 'BMW', 'Ford', 'Nissan', 'Mercedes', 'Jeep']
            },
            'cables': {
                'ru': ['кабель', 'соединение', 'подключение', 'провод', 'разъем', 'распиновка'],
                'uk': ['кабель', "з'єднання", 'підключення', 'провід', "роз'єм", 'розпіновка'],
                'en': ['cable', 'connection', 'wiring', 'connector', 'pinout', 'flexray']
            },
            'tools': {
                'ru': ['ключ', 'инструмент', 'штифт', 'диаметр', 'аксессуары'],
                'uk': ['ключ', 'інструмент', 'штифт', 'діаметр', 'аксесуари'],
                'en': ['key', 'tool', 'pin', 'diameter', 'accessories']
            },
            'support': {
                'ru': ['не работает', 'курс', 'гарантия', 'контакты', 'обучение', 'помощь', 'поддержка', 'что делать'],
                'uk': ['не працює', 'курс', 'гарантія', 'контакти', 'навчання', 'допомога', 'підтримка', 'що робити'],
                'en': ['not working', 'course', 'warranty', 'contacts', 'training', 'help', 'support', 'what to do']
            },
            # 'programming': {
            #     'ru': ['скрипт', 'программирование', 'код', 'программа'],
            #     'uk': ['скрипт', 'програмування', 'код', 'програма'],
            #     'en': ['script', 'programming', 'code', 'program', 'software']
            # }
        }
    
    def detect_product_codes(self, query: str) -> Dict[str, List[str]]:
        """Detect product codes in query and categorize them"""
        query_upper = query.upper()
        detected = {
            'equipment': [],
            'tools': [],
            'unknown': []
        }
        
        # Create regex patterns for different code formats
        patterns = [
            r'MS\d{3}[A-Z]*(?:\s+COM)?',  # MS### or MS###A format, with optional " COM"
            r'MS\d{4,5}-[A-Z]{3}',  # MS####-XXX or MS#####-XXX format (tools)
            r'MS\d{4}[A-Z]*-[A-Z]{3}',  # MS####A-XXX format (tools with letter suffix)
            r'KIT\d{3}[A-Z]*',  # KIT### format
            r'LOKI'  # Special case
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query_upper)
            for match in matches:
                # Clean up the match (remove spaces, normalize)
                clean_match = match.replace(' ', '')
                
                # Check against equipment codes first
                if clean_match in self.equipment_codes:
                    detected['equipment'].append(clean_match)
                elif clean_match in self.tool_codes:
                    detected['tools'].append(clean_match)
                else:
                    # Check if it matches the pattern but with space (like "MS002 COM")
                    if ' COM' in match:
                        base_code = match.replace(' COM', 'COM')
                        if base_code in self.equipment_codes:
                            detected['equipment'].append(base_code)
                        else:
                            detected['unknown'].append(clean_match)
                    else:
                        detected['unknown'].append(clean_match)
        
        return detected
    
    def detect_domain_keywords(self, query: str) -> Dict[str, int]:
        """Count keyword matches for each domain"""
        query_lower = query.lower()
        domain_scores = {}
        
        for domain, languages in self.domain_keywords.items():
            score = 0
            for lang_keywords in languages.values():
                for keyword in lang_keywords:
                    if keyword.lower() in query_lower:
                        score += 1
            domain_scores[domain] = score
        
        return domain_scores
    
    def detect_oem_numbers(self, query: str) -> List[str]:
        """Detect OEM part numbers in query"""
        # Pattern for OEM numbers like 5190161AK, 1501256-00-G
        oem_patterns = [
            r'\d{7}[A-Z]{2}',  # 7 digits + 2 letters
            r'\d{7}-\d{2}-[A-Z]',  # ####-##-# format
            r'\d{4}-\d{3}-\d{2}-[A-Z]'  # ####-###-##-# format
        ]
        
        oem_numbers = []
        for pattern in oem_patterns:
            matches = re.findall(pattern, query)
            oem_numbers.extend(matches)
        
        return oem_numbers
    
    def detect_car_brands(self, query: str) -> List[str]:
        """Detect car brand mentions"""
        car_brands = [
            'BMW', 'Mercedes', 'Ford', 'Nissan', 'Jeep', 'Land Rover', 'Evoque',
            'Wrangler', 'Focus', 'Explorer', 'Ranger', 'Leaf'
        ]
        
        query_lower = query.lower()
        detected_brands = []
        
        for brand in car_brands:
            if brand.lower() in query_lower:
                detected_brands.append(brand)
        
        return detected_brands

class QueryRouter:
    def __init__(self):
        self.detector = ProductCodeDetector()
    
    def analyze_query(self, query: str) -> Dict:
        """Comprehensive query analysis"""
        # Detect product codes
        product_codes = self.detector.detect_product_codes(query)
        
        # Detect domain keywords
        domain_scores = self.detector.detect_domain_keywords(query)
        
        # Detect OEM numbers
        oem_numbers = self.detector.detect_oem_numbers(query)
        
        # Detect car brands
        car_brands = self.detector.detect_car_brands(query)
        
        # Determine primary domains based on scores and detections
        primary_domains = []
        
        # Add domains based on product codes detected
        if product_codes['equipment']:
            primary_domains.append('equipment')
        if product_codes['tools']:
            primary_domains.append('tools')
        
        # Add domains based on keyword scores (threshold: 1+)
        for domain, score in domain_scores.items():
            if score > 0 and domain not in primary_domains:
                primary_domains.append(domain)
        
        # Special rules for compatibility domain
        if oem_numbers or car_brands or any(keyword in query.lower() for keyword in 
            ['генератор от', 'генератор від', 'компрессор с', 'компресор з', 'generator from']):
            if 'compatibility' not in primary_domains:
                primary_domains.append('compatibility')
        
        # If no clear domain detected, route to support
        if not primary_domains:
            primary_domains = ['support']
        
        return {
            'primary_domains': primary_domains,
            'product_codes': product_codes,
            'domain_scores': domain_scores,
            'oem_numbers': oem_numbers,
            'car_brands': car_brands,
            'routing_confidence': self._calculate_confidence(domain_scores, product_codes)
        }
    
    def _calculate_confidence(self, domain_scores: Dict, product_codes: Dict) -> float:
        """Calculate routing confidence score"""
        total_keywords = sum(domain_scores.values())
        total_codes = len(product_codes['equipment']) + len(product_codes['tools'])
        
        if total_codes > 0:
            return min(0.9, 0.5 + (total_codes * 0.2) + (total_keywords * 0.1))
        elif total_keywords > 2:
            return min(0.8, 0.4 + (total_keywords * 0.1))
        elif total_keywords > 0:
            return 0.6
        else:
            return 0.3

class PromptStaticAnalyzer:
    def __init__(self):
        self.router = QueryRouter()
        self.orchestrator_prompt = """
        You are an intelligent orchestrator for a multi-assistant customer support system.
        Your role is to analyze user queries and route them to the most appropriate specialist assistant.
        """
    
    def route_query(self, user_query: str) -> Dict:
        """Main routing function that combines analysis with LLM routing"""
        
        # Step 1: Analyze query with Python detection
        analysis = self.router.analyze_query(user_query)
        
        # Step 2: Create enhanced context for LLM
        enhanced_context = self._create_enhanced_context(user_query, analysis)
        
        # Step 3: Get LLM routing decision (mock for now)
        keyword_routing = self._get_keyword_routing(enhanced_context)
        
        # Step 4: Combine Python analysis with LLM decision
        final_routing = self._combine_routing_decisions(analysis, keyword_routing)
        
        return final_routing
    
    def _create_enhanced_context(self, query: str, analysis: Dict) -> str:
        """Create enhanced context for the LLM with detected information"""
        
        context_parts = [
            f"USER QUERY: {query}",
            "",
            "=== DETECTED INFORMATION ==="
        ]

        # Add detected product codes
        if analysis['primary_domains']:
            context_parts.append(f"Primary domains assigned by static analyzer: {', '.join(analysis['primary_domains'])}")
        
        # Add detected product codes
        if analysis['product_codes']['equipment']:
            context_parts.append(f"Equipment codes detected: {', '.join(analysis['product_codes']['equipment'])}")
        
        if analysis['product_codes']['tools']:
            context_parts.append(f"Tool codes detected: {', '.join(analysis['product_codes']['tools'])}")
        
        # Add OEM numbers
        if analysis['oem_numbers']:
            context_parts.append(f"OEM numbers detected: {', '.join(analysis['oem_numbers'])}")
        
        # Add car brands
        if analysis['car_brands']:
            context_parts.append(f"Car brands detected: {', '.join(analysis['car_brands'])}")
        
        # Add domain keyword scores
        if analysis['domain_scores']:
            scored_domains = [(domain, score) for domain, score in analysis['domain_scores'].items() if score > 0]
            if scored_domains:
                context_parts.append("Domain keyword matches:")
                for domain, score in sorted(scored_domains, key=lambda x: x[1], reverse=True):
                    context_parts.append(f"  - {domain}: {score} matches")
        
        # Add confidence
        context_parts.append(f"Routing confidence: {analysis['routing_confidence']:.2f}")
        context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _get_keyword_routing(self, enhanced_context: str) -> Dict:
        """Send enhanced context to LLM for routing decision"""
        # This is a mock implementation - replace with actual LLM call
        # For demonstration, we'll use simple keyword-based routing
        
        context_lower = enhanced_context.lower()
        specialists = []
        reasoning = ""
        
        if any(word in context_lower for word in ['equipment', 'стенд', 'ms']):
            specialists.append('equipment')
            reasoning += "Equipment-related query detected. "
        
        if any(word in context_lower for word in ['test', 'check', 'проверить', 'перевірити']):
            specialists.append('diagnostics')
            reasoning += "Testing procedure query detected. "
        
        if any(word in context_lower for word in ['oem', 'car brands', 'bmw', 'ford', 'mercedes']):
            specialists.append('compatibility')
            reasoning += "Vehicle compatibility query detected. "
        
        if any(word in context_lower for word in ['cable', 'кабель']):
            specialists.append('cables')
            reasoning += "Cable-related query detected. "
        
        if any(word in context_lower for word in ['key', 'tool', 'ключ']):
            specialists.append('tools')
            reasoning += "Tool-related query detected. "
        
        if any(word in context_lower for word in ['not working', 'не работает', 'warranty', 'course']):
            specialists.append('support')
            reasoning += "Support issue detected. "
        
        # if any(word in context_lower for word in ['script', 'скрипт', 'programming']):
        #     specialists.append('programming')
        #     reasoning += "Programming query detected. "
        
        if not specialists:
            specialists = ['support']
            reasoning = "No specific domain detected, routing to support."
        
        return {
            "specialists": specialists,
            "reasoning": reasoning.strip(),
            "confidence": 0.85
        }
    
    def _combine_routing_decisions(self, analysis: Dict, keyword_routing: Dict) -> Dict:
        """Combine Python analysis with LLM routing decision"""
        
        # Start with LLM decision
        final_specialists = set(keyword_routing.get("specialists", []))
        
        # Add specialists based on product code detection
        if analysis['product_codes']['equipment']:
            final_specialists.add('equipment')
        
        if analysis['product_codes']['tools']:
            final_specialists.add('tools')
        
        # Add compatibility if OEM/car brands detected
        if analysis['oem_numbers'] or analysis['car_brands']:
            final_specialists.add('compatibility')
        
        # Ensure at least one specialist is selected
        if not final_specialists:
            final_specialists.add('support')
        
        return {
            "specialists": sorted(list(final_specialists)),
            "analysis": analysis,
            "keyword_routing": keyword_routing,
            "final_confidence": min(analysis['routing_confidence'], keyword_routing.get('confidence', 0.5)),
            "routing_context": self._generate_specialist_context(analysis, list(final_specialists))
        }
    
    def _generate_specialist_context(self, analysis: Dict, specialists: List[str]) -> str:
        """Generate context to send to specialist(s)"""
        
        context_parts = [
            "=== CONTEXT FOR SPECIALIST ===",
            f"Chosen Specialists: {', '.join(specialists)}",
            ""
        ]
        
        if analysis['product_codes']['equipment'] or analysis['product_codes']['tools']:
            context_parts.append("Detected Product Codes:")
            if analysis['product_codes']['equipment']:
                context_parts.append(f"  Equipment: {', '.join(analysis['product_codes']['equipment'])}")
            if analysis['product_codes']['tools']:
                context_parts.append(f"  Tools: {', '.join(analysis['product_codes']['tools'])}")
            context_parts.append("")
        
        if analysis['oem_numbers']:
            context_parts.append(f"OEM Numbers: {', '.join(analysis['oem_numbers'])}")
        
        if analysis['car_brands']:
            context_parts.append(f"Vehicle Brands: {', '.join(analysis['car_brands'])}")
        
        context_parts.extend([
            "",
            f"Routing Confidence: {analysis['routing_confidence']:.2f}",
            "=== END CONTEXT ==="
        ])
        
        return "\n".join(context_parts)

# Usage example
if __name__ == "__main__":
    orchestrator = PromptStaticAnalyzer()
    
    # Test the complete flow
    test_queries = [
        "Какой стенд для генераторов ты посоветуешь для маленькой мастерской?",
        "как перевірити діодний міст тестером MS021?",
        "Генератор от Jeep Wrangler 5190161AK можно проверить на MS005A?",
        "Який кабель для стенда MS112 для компресора Mercedes мені потрібен?",
        "У меня не работает оборудование MS800, что мне нужно сделать?",
        "чем отличается MS800 от MS800A? можно предоставить в виде таблицы?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"USER QUERY: {query}")
        print('='*60)
        
        result = orchestrator.route_query(query)
        
        print(f"ROUTED TO: {', '.join(result['specialists'])}")
        print(f"CONFIDENCE: {result['final_confidence']:.2f}")
        print(f"\nCONTEXT FOR SPECIALIST(S):")
        print(result['routing_context'])
        print(f"\nDETAILED ANALYSIS:")
        print(json.dumps(result['analysis'], indent=2, ensure_ascii=False))