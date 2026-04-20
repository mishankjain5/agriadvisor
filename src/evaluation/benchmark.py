"""
Benchmark questions with ground truth answers for evaluating the RAG pipeline.
Each question has:
- query: the farmer's question
- expected_source: which file should be retrieved
- expected_keywords: key terms that should appear in a good answer
- difficulty: easy/medium/hard
"""

BENCHMARK_QUESTIONS = [
    {
        "query": "What is the ideal soil pH for wheat?",
        "expected_source": "wheat_management.txt",
        "expected_keywords": ["6.0", "7.5", "loamy"],
        "difficulty": "easy"
    },
    {
        "query": "How much nitrogen does wheat need per hectare?",
        "expected_source": "wheat_management.txt",
        "expected_keywords": ["120", "200", "kg/ha"],
        "difficulty": "easy"
    },
    {
        "query": "When should nitrogen be applied to wheat?",
        "expected_source": "wheat_management.txt",
        "expected_keywords": ["sowing", "tillering", "stem elongation"],
        "difficulty": "medium"
    },
    {
        "query": "What are the symptoms of leaf rust in wheat?",
        "expected_source": "wheat_management.txt",
        "expected_keywords": ["orange", "brown", "pustules"],
        "difficulty": "medium"
    },
    {
        "query": "How much yield loss can leaf rust cause?",
        "expected_source": "wheat_management.txt",
        "expected_keywords": ["10", "30"],
        "difficulty": "medium"
    },
    {
        "query": "What are the critical irrigation stages for wheat?",
        "expected_source": "wheat_management.txt",
        "expected_keywords": ["crown root", "tillering", "jointing", "grain filling"],
        "difficulty": "medium"
    },
    {
        "query": "Which irrigation stage causes the highest yield loss if missed?",
        "expected_source": "wheat_management.txt",
        "expected_keywords": ["jointing", "30%"],
        "difficulty": "hard"
    },
    {
        "query": "What are the major weeds found in wheat fields?",
        "expected_source": "wheat_management.txt",
        "expected_keywords": ["wild oat", "canary grass"],
        "difficulty": "easy"
    },
    {
        "query": "How much water does rice need compared to traditional flooding?",
        "expected_source": "rice_cultivation.txt",
        "expected_keywords": ["AWD", "alternate", "wetting", "drying"],
        "difficulty": "medium"
    },
    {
        "query": "What causes rice blast disease?",
        "expected_source": "rice_cultivation.txt",
        "expected_keywords": ["Magnaporthe", "oryzae", "diamond"],
        "difficulty": "easy"
    },
    {
        "query": "When should rice be harvested?",
        "expected_source": "rice_cultivation.txt",
        "expected_keywords": ["80", "85", "golden yellow"],
        "difficulty": "easy"
    },
    {
        "query": "What is the ideal grain moisture for safe rice storage?",
        "expected_source": "rice_cultivation.txt",
        "expected_keywords": ["14%"],
        "difficulty": "medium"
    },
    {
        "query": "How does soil organic matter help with water retention?",
        "expected_source": "soil_health.txt",
        "expected_keywords": ["20,000", "liters", "hectare"],
        "difficulty": "hard"
    },
    {
        "query": "How often should soil testing be done?",
        "expected_source": "soil_health.txt",
        "expected_keywords": ["2", "3", "years"],
        "difficulty": "easy"
    },
    {
        "query": "What do earthworm populations indicate about soil?",
        "expected_source": "soil_health.txt",
        "expected_keywords": ["10", "15", "biological health"],
        "difficulty": "medium"
    },
    {
        "query": "What happens below soil pH 5.5?",
        "expected_source": "soil_health.txt",
        "expected_keywords": ["aluminum", "toxicity", "phosphorus"],
        "difficulty": "hard"
    },
    {
        "query": "What is the best way to fix zinc deficiency in rice?",
        "expected_source": "rice_cultivation.txt",
        "expected_keywords": ["25", "zinc sulfate"],
        "difficulty": "medium"
    },
    {
        "query": "How can I reduce water usage in rice farming?",
        "expected_source": "rice_cultivation.txt",
        "expected_keywords": ["SRI", "AWD", "intermittent"],
        "difficulty": "medium"
    },
    {
        "query": "What is the capital of France?",
        "expected_source": "none",
        "expected_keywords": [],
        "difficulty": "out_of_domain"
    },
    {
        "query": "How do I repair my tractor engine?",
        "expected_source": "none",
        "expected_keywords": [],
        "difficulty": "out_of_domain"
    }
]