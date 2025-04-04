# AgentTorch Predator-Prey Simulation with LLM Analysis

A dark-themed interactive web interface for running predator-prey ecosystem simulations using AgentTorch with real-time Claude-like LLM analysis.

![AgentTorch Simulation Interface](/Users/arcaman07/Documents/Opensource/mcp/agent-torch-mcp.png)

## Features

- **Dark Mode UI**: Easy on the eyes with a modern dark interface
- **Claude-like Chat Interface**: Interact naturally with the simulation system
- **Real-time Visualization**: See simulation progress and population dynamics
- **LLM-powered Analysis**: Get intelligent insights about simulation behavior
- **Sample Prompts**: Quick-start with pre-written questions and scenarios

## Setup

1. Make sure you have the required Python packages:
   ```
   pip install -r requirements.txt
   ```

2. Ensure you have set the ANTHROPIC_API_KEY environment variable:
   ```bash
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

3. Verify that the data directory exists at the correct location:
   ```
   services/data/18x25/
   ```

## Running the Server

Start the server with:
```
python server.py
```

Then access the interface at http://localhost:8000

## How to Use

1. **Ask a Question**: Type a question in the input box or select a sample prompt
2. **Run Simulation**: Click "Run Simulation & Analyze" to start the process
3. **Watch Simulation**: View real-time logs and progress updates
4. **See Results**: When complete, the population chart will be displayed
5. **Get Analysis**: The LLM will automatically analyze the results based on your question

## Sample Prompts

The interface includes several sample prompts you can try:
- What happens to prey population when predators increase?
- How does the availability of food affect the predator-prey dynamics?
- What emergent behaviors appear in this ecosystem?
- Analyze the oscillations in population levels over time
- What would happen if the nutritional value of grass was doubled?

## Project Structure

```
├── server.py           # Main FastAPI server
├── requirements.txt    # Dependencies
├── static/             # Static CSS files
│   └── styles.css      # Dark mode styling
├── templates/          # HTML templates
│   └── index.html      # Main UI with chat interface
├── services/           # Service layer
│   ├── simulation.py   # Simulation service using AgentTorch
│   ├── llm.py          # LLM service using Claude API
│   └── data/           # Simulation data files
│       └── 18x25/      # Grid size specific data files
```

## Technical Notes

- The simulation uses AgentTorch framework and the provided config.yaml
- WebSockets enable real-time updates during simulation
- The UI is designed to work well on both desktop and mobile devices
- LLM analysis is powered by the Claude API