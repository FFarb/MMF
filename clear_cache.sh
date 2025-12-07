#!/bin/bash
# Clear Python cache and restart training

echo "Clearing Python bytecode cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

echo "Verifying git status..."
git status

echo "Pulling latest changes..."
git pull

echo "Cache cleared. Please restart your training script."
