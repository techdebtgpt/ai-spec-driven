#!/bin/bash
# Spec-Agent Initialization Script
# This script sets up the development environment for the spec-driven-development-agent

set -e  # Exit on error

echo "Initializing Spec-Driven Development Agent..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $PYTHON_VERSION"
echo ""

# Create virtual environment if it doesn't exist
if [ -d ".venv" ]; then
    echo "Virtual environment already exists, skipping creation..."
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "pip upgraded"
echo ""

# Install the package with dev dependencies
echo "Installing spec-agent with dev dependencies..."
pip install -e ".[dev]"
echo "Installation complete"
echo ""

# Verify installation
echo "Verifying installation..."
if command -v spec-agent &> /dev/null; then
    echo "spec-agent command is available"
    echo ""
    echo "Setup complete! You can now use spec-agent."
    echo ""
    echo "To activate the virtual environment in your current shell, run:"
    echo "  source .venv/bin/activate"
    echo ""
    echo "To see available commands, run:"
    echo "  spec-agent --help"
    echo ""
    echo "Example usage:"
    echo "  spec-agent start /path/to/repo --branch main --description \"Your task description\""
else
    echo "Warning: spec-agent command not found. You may need to activate the venv:"
    echo "  source .venv/bin/activate"
fi

