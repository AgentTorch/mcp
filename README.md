# AgentTorch MCP Server

A Model-Controlled Process (MCP) server that enables natural language-driven simulations of Antarctic ecosystems with emperor penguins and leopard seals.

![AgentTorch MCP Screenshot](static/images/antarctica_simulation.png)

## Overview

AgentTorch MCP Server provides a web interface for running large-scale agent-based simulations through natural language queries. It uses Claude AI to interpret user requests, extract simulation parameters, and generate analysis of results.

Key features:
- Run simulations with up to 10,000 agents (9,000 emperor penguins and 1,000 leopard seals)
- Natural language interface using Anthropic's Claude
- Real-time visualization of population dynamics
- Detailed simulation logs for tracking ecosystem behavior
- Add custom behaviors through natural language descriptions

## Requirements

- Python 3.9+
- PyTorch 2.0+
- FastAPI
- Anthropic API key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/agent-torch-mcp.git
   cd agent-torch-mcp
Install dependencies:
bash
pip install -r requirements.txt
Set your Anthropic API key:
bash
# For Linux/macOS
export ANTHROPIC_API_KEY=your_anthropic_api_key_here

# For Windows (Command Prompt)
set ANTHROPIC_API_KEY=your_anthropic_api_key_here

# For Windows (PowerShell)
$env:ANTHROPIC_API_KEY="your_anthropic_api_key_here"
Create necessary directories:
bash
mkdir -p static/images
mkdir -p static/css
Running the Server
Start the server with:
bash
python app.py
Visit http://localhost:8000 in your browser to access the web interface.
Project Structure
agent-torch-mcp/
├── app.py                  # FastAPI main application entry point
├── controller.py           # Central controller coordinating services
├── requirements.txt        # Dependencies list for pip installation
├── README.md               # Project documentation
├── .env.example            # Example environment variables
├── .gitignore              # Standard Git ignore file
│
├── config/                 # Configuration files
│   └── predator_prey/
│       └── config.yaml     # Predator-prey simulation config
│
├── static/                 # Static files for web interface
│   ├── css/
│   │   └── styles.css
│   └── images/             # Generated visualizations
│
├── models/                 # Data models
│   ├── __init__.py
│   ├── nlp.py              # Schema models for NLP and chat requests
│   └── simulation.py       # Schema models for simulation results
│
├── routers/                # API route handlers
│   ├── __init__.py
│   ├── chat.py             # Endpoints for chat functionality
│   └── simulation.py       # Endpoints for simulation requests
│
├── services/               # Core business logic
│   ├── __init__.py
│   ├── llm.py              # Claude/LLM interface service
│   ├── nlp_parser.py       # Natural language parsing service
│   └── simulation.py       # Simulation execution service
│
└── templates/              # Jinja2 templates
    ├── index.html          # Main web interface
    └── error.html          # Error page template
