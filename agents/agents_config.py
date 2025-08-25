assistant_configs = {
    "equipment": {
        "id": "asst_awHzb3xl9uRV0SvuLjOteTus",
        "purpose": "Equipment expert",
        "truncation_strategy": {
            "type": "last_messages",
            "last_n_messages": 6
        },
        "max_prompt_tokens": None,
        "max_completion_tokens": None
    },

    "diagnostics": {
        "id": "asst_II6EhFFETHGmydpVxGrOVzQG",
        "purpose": "Diagnostics expert",
        "truncation_strategy": {
            "type": "last_messages",
            "last_n_messages": 6
        },
        "max_prompt_tokens": None,
        "max_completion_tokens": None
    },

    "tools": {
        "id": "asst_XVvRqM5N2gtZNafuQFCQ4Gbe",
        "purpose": "Hand tools expert",
        "truncation_strategy": {
            "type": "last_messages",
            "last_n_messages": 6
        },
        "max_prompt_tokens": None,
        "max_completion_tokens": None
    },

    "cables": {
        "id": "asst_AEIIYE4ElLmYqjo5eBJdXssm",
        "purpose": "Cables for equipment connection expert",
        "truncation_strategy": {
            "type": "last_messages",
            "last_n_messages": 6
        },
        "max_prompt_tokens": None,
        "max_completion_tokens": None
    },

    "support": {
        "id": "asst_gRr26pcbjcrcBz6e1N2bb2y9",
        "purpose": "Expert with common information knowledge",
        "truncation_strategy": {
            "type": "last_messages",
            "last_n_messages": 6
        },
        "max_prompt_tokens": None,
        "max_completion_tokens": None
    },

    "scripts": {
        "id": "asst_494Ciy6OSsnrbIU99G0pcW76",
        "purpose": "Expert in creating scripts for MS005-MS005A",
        "truncation_strategy": {
            "type": "last_messages",
            "last_n_messages": 6
        },
        "max_prompt_tokens": None,
        "max_completion_tokens": None
    }
}

price_per_token_in = 2.5/1000_000
price_per_token_out = 10/1000_000