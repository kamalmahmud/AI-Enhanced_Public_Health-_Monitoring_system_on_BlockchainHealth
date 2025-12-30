@echo off
echo Starting Government Epidemic Monitoring System...

echo Starting Blockchain...
start cmd /k "cd /d C:\dev\myblockchain && npx hardhat node"

timeout /t 4

echo Starting IPFS...
start cmd /k "ipfs daemon"

timeout /t 4

echo Starting Mistral AI...
start cmd /k "ollama run mistral"

timeout /t 4

echo Starting AI Agent...
start cmd /k "cd /d C:\dev\ai && python ai_agent.py"
