assistant_configs = {
    "orchestrator": {
        "id": "asst_aU6DIODwxNlFRxrY3WipBPjz",
        "purpose": "Route requests and coordinate other assistants",
        "truncation_strategy": {
            "type": "last_messages",
            "last_n_messages": 20
        },
        "max_prompt_tokens": None,
        "max_completion_tokens": None
    },

    "equipment": {
        "id": "asst_1dQLsAz9p6T2cQyGtnjSeXnv",
        "purpose": "Equipment expert",
        "truncation_strategy": {
            "type": "last_messages",
            "last_n_messages": 20
        },
        "max_prompt_tokens": None,
        "max_completion_tokens": None
    },

    "tools": {
        "id": "asst_jtOdIxiHK1UsVkXaCxM8y0PS",
        "purpose": "Tools expert",
        "truncation_strategy": {
            "type": "last_messages",
            "last_n_messages": 20
        },
        "max_prompt_tokens": None,
        "max_completion_tokens": None
    },

    "cables": {
        "id": "asst_cErO4m6RZdfHQPAT3wVagp2z",
        "purpose": "Cables for equipment connection expert",
        "truncation_strategy": {
            "type": "last_messages",
            "last_n_messages": 20
        },
        "max_prompt_tokens": None,
        "max_completion_tokens": None
    },

    "commonInfo": {
        "id": "asst_nJzNpbdII7UzbOGiiSFcu09u",
        "purpose": "Expert with common information knowledge",
        "truncation_strategy": {
            "type": "last_messages",
            "last_n_messages": 20
        },
        "max_prompt_tokens": None,
        "max_completion_tokens": None
    },

    "combinator": {
        "id": "asst_FM5jrNCeRHxy3MpMueV1RkED",
        "purpose": "Combine experts responses into a final response for user",
        "truncation_strategy": {
            "type": "last_messages",
            "last_n_messages": 20
        },
        "max_prompt_tokens": None,
        "max_completion_tokens": None
    }
}

price_per_token_in = 2.5/1000_000
price_per_token_out = 10/1000_000