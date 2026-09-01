#!/bin/bash
# setup.sh - For Streamlit Cloud deployment

mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = true\n\
" > ~/.streamlit/config.toml

echo "Setup complete!"
