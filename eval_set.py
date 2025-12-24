EVAL_QUESTIONS = [

    # --- BASIC FACT RETRIEVAL ---
    {
        "question": "What are the two main types of electric vehicles discussed?",
        "source_hint": "All-electric vehicles, also referred to as battery electric vehicles"
    },

    {
        "question": "What is regenerative braking and what does it do?",
        "source_hint": "Regenerative braking allows EVs to capture energy normally lost during braking"
    },

    # --- PARAPHRASE / SEMANTIC MATCH ---
    {
        "question": "How far can most modern battery electric vehicles travel on a single charge?",
        "source_hint": "Most new BEVs are designed to travel between 110 and over 300 miles"
    },

    {
        "question": "Why does cold or hot weather reduce EV driving range?",
        "source_hint": "Extreme temperatures tend to reduce range because energy from the battery powers climate control systems"
    },

    # --- MULTI-SENTENCE / CONTEXTUAL ---
    {
        "question": "How do plug-in hybrid electric vehicles switch between electric and gasoline power?",
        "source_hint": "The engine will then power on when the battery is mostly depleted"
    },

    {
        "question": "What factors influence how long it takes to charge an EV?",
        "source_hint": "charging times vary based on the type or level of charging"
    },

    # --- COMPARATIVE / TABLE-BASED ---
    {
        "question": "How do emissions from all-electric vehicles compare to conventional vehicles?",
        "source_hint": "All-electric vehicles produce zero tailpipe emissions"
    },

    {
        "question": "What are the typical charging speeds for Level 1 and Level 2 charging?",
        "source_hint": "Level 1 2–5 miles of range per hour"
    },

    # --- COST / NUMERICAL REASONING ---
    {
        "question": "How much can EV drivers save in fuel costs over 15 years?",
        "source_hint": "$14,500 in fuel costs"
    },

    {
        "question": "What is the typical cost range for installing a Level 2 charger?",
        "source_hint": "Level 2** 10–30 miles of range per hour"
    },

    # --- NEGATIVE / SHOULD FAIL ---
    {
        "question": "What is the maximum towing capacity of electric vehicles discussed here?",
        "source_hint": None  # Not in document
    },

    {
        "question": "Which electric vehicles use nuclear power?",
        "source_hint": None  # Not in document
    },
]
