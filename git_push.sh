#!/bin/bash

# Change to your repository directory (optional if you run inside it)
cd /home/hogu/Desktop/savi-2025-2026-trabalho1-grupo7

# Add all changes
git add .

# Commit (auto message, or blank)
git commit -am "Auto commit"


# Safely pull remote changes before pushing
git pull --rebase origin main

# Push to origin
git push

echo "Git push done!"

