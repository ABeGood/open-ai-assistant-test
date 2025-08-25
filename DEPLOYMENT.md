# Railway Deployment Guide

This guide explains how to deploy the OpenAI Assistant customer support system to Railway.

## Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **GitHub Repository**: Your code should be pushed to a GitHub repository
3. **API Keys**:
   - OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
   - Telegram Bot token from [BotFather](https://t.me/botfather)

## Deployment Steps

### 1. Connect Repository to Railway

1. Go to [railway.app](https://railway.app) and sign in
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repository containing this project
5. Railway will automatically detect the Dockerfile

### 2. Configure Environment Variables

In the Railway dashboard, go to your project settings and add these environment variables:

```
TELEGRAM_TOKEN=your_telegram_bot_token_here
OPENAI_TOKEN=your_openai_api_key_here
```

### 3. Deploy

1. Railway will automatically start building and deploying your application
2. The deployment uses the provided `Dockerfile` and `railway.toml` configuration
3. Your bot will be running 24/7 once deployed

## Files Created for Deployment

- **`Dockerfile`**: Containerizes the Python application
- **`railway.toml`**: Railway-specific configuration
- **`requirements_clean.txt`**: Clean Python dependencies (removes Windows-specific packages)
- **`.dockerignore`**: Excludes unnecessary files from Docker build
- **`DEPLOYMENT.md`**: This documentation file

## Application Structure

The application uses:
- **Main Entry Point**: `main.py` - Starts the Telegram bot
- **Architecture**: Multi-agent system with orchestrator and specialist assistants
- **Dependencies**: OpenAI API, python-telegram-bot, pydantic, requests

## Environment Variables Required

| Variable | Description | Example |
|----------|-------------|---------|
| `TELEGRAM_TOKEN` | Bot token from BotFather | `123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11` |
| `OPENAI_TOKEN` | OpenAI API key | `sk-proj-abc123...` |

## Monitoring and Logs

1. Access logs through Railway dashboard
2. Monitor application health and resource usage
3. View real-time deployment status

## Troubleshooting

### Common Issues

1. **Environment Variables**: Ensure both `TELEGRAM_TOKEN` and `OPENAI_TOKEN` are set
2. **Build Failures**: Check that all required files are included and not in `.dockerignore`
3. **Runtime Errors**: Check Railway logs for detailed error messages

### Local Testing

Before deploying, test locally:

```bash
# Create .env file with your tokens
echo "TELEGRAM_TOKEN=your_token" > .env
echo "OPENAI_TOKEN=your_key" >> .env

# Install dependencies
pip install -r requirements_clean.txt

# Run locally
python main.py
```

## Scaling

Railway automatically handles:
- Container restarts on failure
- Resource allocation
- SSL/TLS certificates
- Domain generation

The application is designed to handle multiple concurrent users through the OpenAI Assistant API's built-in session management.