#!/bin/bash
# setup.sh: opens the Study Prompter in Google Chrome.
# No dependencies to install, just a single HTML file.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HTML_FILE="$SCRIPT_DIR/study-prompter.html"

if [[ ! -f "$HTML_FILE" ]]; then
  echo "Error: study-prompter.html not found in $SCRIPT_DIR"
  exit 1
fi

echo "Opening Study Prompter..."

echo ""

if [[ "$OSTYPE" == "darwin"* ]]; then
  open -a "Google Chrome" "$HTML_FILE" 2>/dev/null || open "$HTML_FILE"
elif command -v google-chrome &>/dev/null; then
  google-chrome "$HTML_FILE" &
elif command -v xdg-open &>/dev/null; then
  xdg-open "$HTML_FILE" &
else
  echo "Open this file in Chrome: $HTML_FILE"
fi
