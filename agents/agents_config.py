assistant_configs = {
    "equipment": {
        "id": "asst_1dQLsAz9p6T2cQyGtnjSeXnv",
        "purpose": "Equipment expert",
        "truncation_strategy": {
            "type": "last_messages",
            "last_n_messages": 1
        },
        "max_prompt_tokens": None,
        "max_completion_tokens": None
    },

    "diagnostics": {
        "id": "asst_II6EhFFETHGmydpVxGrOVzQG",
        "purpose": "Diagnostics expert",
        "truncation_strategy": {
            "type": "last_messages",
            "last_n_messages": 1
        },
        "max_prompt_tokens": None,
        "max_completion_tokens": None
    },

    "tools": {
        "id": "asst_jtOdIxiHK1UsVkXaCxM8y0PS",
        "purpose": "Hand tools expert",
        "truncation_strategy": {
            "type": "last_messages",
            "last_n_messages": 1
        },
        "max_prompt_tokens": None,
        "max_completion_tokens": None
    },

    "cables": {
        "id": "asst_cErO4m6RZdfHQPAT3wVagp2z",
        "purpose": "Cables for equipment connection expert",
        "truncation_strategy": {
            "type": "last_messages",
            "last_n_messages": 1
        },
        "max_prompt_tokens": None,
        "max_completion_tokens": None
    },

    "support": {
        "id": "asst_nJzNpbdII7UzbOGiiSFcu09u",
        "purpose": "Expert with common information knowledge",
        "truncation_strategy": {
            "type": "last_messages",
            "last_n_messages": 1
        },
        "max_prompt_tokens": None,
        "max_completion_tokens": None
    },

    "scripts": {
        "id": "asst_494Ciy6OSsnrbIU99G0pcW76",
        "purpose": "Expert in creating scripts for MS005-MS005A",
        "truncation_strategy": {
            "type": "last_messages",
            "last_n_messages": 1
        },
        "max_prompt_tokens": None,
        "max_completion_tokens": None
    }
}

price_per_token_in = 2.5/1000_000
price_per_token_out = 10/1000_000