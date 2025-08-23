"""
Main entry point for the customer support system.
Configures and starts the Telegram bot with orchestrator integration.
"""

import os
from dotenv import load_dotenv

from agents.orchestrator import create_orchestrator
from telegram_bot.bot import TelegramBot


def main():
    """Initialize and run the customer support bot."""
    load_dotenv()
    
    telegram_token = os.environ.get("TELEGRAM_TOKEN_OLD")
    if not telegram_token:
        raise ValueError("TELEGRAM_TOKEN_OLD environment variable is required")
    
    # Create orchestrator
    orchestrator = create_orchestrator()
    
    # Initialize and start bot
    bot = TelegramBot(bot_token=telegram_token, orchestrator=orchestrator)
    bot.run()


if __name__ == "__main__":
    main()