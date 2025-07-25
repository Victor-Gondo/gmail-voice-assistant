# Gmail Voice Assistant

A Python-based voice assistant that integrates with Gmail using the Gmail API and MCP (Model Context Protocol) servers.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+**
- **Node.js and npm** (for npx)
- **Google Cloud Project** with Gmail API enabled

## 🚀 Installation

### 1. Set Up Virtual Environment (Recommended)

It's highly recommended to use a virtual environment to isolate your project dependencies.

#### Option A: Using venv (Python's built-in virtual environment)

```bash
# Create a virtual environment
python -m venv gmail-voice-env

# Activate the virtual environment
# On macOS/Linux:
source gmail-voice-env/bin/activate
# On Windows:
gmail-voice-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using Conda

```bash
# Create a conda environment
conda create -n gmail-voice-env python=3.8

# Activate the conda environment
conda activate gmail-voice-env

# Install dependencies
pip install -r requirements.txt
```


### 2. Install Python Dependencies (if not using virtual environment)

```bash
pip install -r requirements.txt
```

### 2. Install Node.js and npm

#### Check if you already have npx:
```bash
npx --version
```

If npx is not available, install Node.js:

**macOS (Homebrew):**
```bash
brew install node
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install nodejs npm
```

**Windows:**
Download the installer from [https://nodejs.org](https://nodejs.org)

## 🔐 Google Cloud Setup

### 1. Enable Gmail API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Navigate to "APIs & Services" > "Library"
4. Search for "Gmail API" and enable it

### 2. Create OAuth 2.0 Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth client ID"
3. Choose application type:
   - **Desktop app** (recommended for local development)
   - **Web application** (if using web interface)
4. Give your credentials a descriptive name
5. For Web application, add `http://localhost:3000/oauth2callback` to authorized redirect URIs
6. Click "Create"
7. Download the JSON credentials file
8. Rename the file to `gcp-oauth.keys.json`

## 🔑 Authentication Setup

### Option 1: Global Authentication (Recommended)

1. Create the authentication directory:
```bash
mkdir -p ~/.gmail-mcp
```

2. Move your credentials file:
```bash
mv gcp-oauth.keys.json ~/.gmail-mcp/
```

3. Install and authenticate:
```bash
npx -y @smithery/cli install @gongrzhe/server-gmail-autoauth-mcp --client claude
npx @gongrzhe/server-gmail-autoauth-mcp auth
```

### Option 2: Local Authentication

1. Place `gcp-oauth.keys.json` in your project directory or in `./.gmail-mcp/`
2. Run the authentication commands:
```bash
npx -y @smithery/cli install @gongrzhe/server-gmail-autoauth-mcp --client claude
npx @gongrzhe/server-gmail-autoauth-mcp auth
```

## 🎯 Usage

Once authentication is complete, run the voice assistant:

**If using a virtual environment, make sure it's activated first:**
```bash
# For venv:
source gmail-voice-env/bin/activate  # macOS/Linux
# or
gmail-voice-env\Scripts\activate     # Windows

# For conda:
conda activate gmail-voice-env
```

**Then run the voice assistant:**
```bash
python agent.py
```

### Deactivating the Virtual Environment

When you're done working on the project:

```bash
# For venv:
deactivate

# For conda:
conda deactivate
```

## 📁 Project Structure

```
gmail-voice-assistant/
├── agent.py              # Main voice assistant script
├── requirements.txt      # Python dependencies
├── mcp_servers.json     # MCP server configuration
├── env.example          # Environment variables example
└── README.md           # This file
```

## 🛠️ Configuration

### Environment Variables

Copy `env.example` to `.env` and configure your environment variables:

```bash
cp env.example .env
```

**Required Variables:**
- `OPENAI_API_KEY`: Your OpenAI API key for speech-to-text and GPT models

**Optional Variables:**
- `OPENAI_MODEL`: OpenAI model to use (default: gpt-4.1)
- `VOICE_SILENCE_THRESHOLD`: Amplitude threshold for silence detection (default: 500)
- `VOICE_SILENCE_DURATION`: Silence duration in seconds (default: 0.2)
- `VAD_AGGRESSIVENESS`: Voice Activity Detection aggressiveness 0-3 (default: 3)
- `ASSISTANT_SYSTEM_PROMPT`: Custom system prompt for the assistant
- `DEBUG`: Enable debug logging (true/false)
- `MCP_LOG_LEVEL`: MCP server log level (debug, info, warn, error)

### Other Configuration Files

- **mcp_servers.json**: Configure MCP servers for Gmail integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

[Add your license information here]

## 🆘 Troubleshooting

### Common Issues

1. **"npx command not found"**
   - Ensure Node.js and npm are properly installed
   - Try reinstalling Node.js

2. **Authentication errors**
   - Verify your `gcp-oauth.keys.json` file is in the correct location
   - Ensure Gmail API is enabled in your Google Cloud project
   - Check that your OAuth credentials are properly configured

3. **Import errors**
   - Run `pip install -r requirements.txt` to install all dependencies
   - Check your Python version (3.8+ required)

### Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Review the Google Cloud Console for API quotas and errors
3. Ensure all prerequisites are met
4. Check the project's issue tracker for known problems
