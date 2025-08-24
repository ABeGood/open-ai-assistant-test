"""
Main entry point for the customer support system.
Configures and starts the Telegram bot with orchestrator integration.
"""

import os
from dotenv import load_dotenv

from telegram_bot.bot import TelegramBot
from openai import OpenAI


def main():
    """Initialize and run the customer support bot."""
    load_dotenv()
    
    telegram_token = os.environ.get("TELEGRAM_TOKEN")
    if not telegram_token:
        raise ValueError("TELEGRAM_TOKEN environment variable is required")
    
    open_ai_token = os.environ.get("OPENAI_TOKEN")
    if not open_ai_token:
        raise ValueError("OPENAI_TOKEN environment variable is required")


    open_ai_client = OpenAI(api_key=open_ai_token, timeout=120, max_retries=3)
    
    # Initialize and start bot
    bot = TelegramBot(bot_token=telegram_token, llm_client=open_ai_client)
    bot.run()


if __name__ == "__main__":
    main()