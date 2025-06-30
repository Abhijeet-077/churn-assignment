mkdir -p ~/.streamlit/

echo "\
[theme]\n\
base = 'dark'\n\
primaryColor = '#00ffff'\n\
backgroundColor = '#0d1117'\n\
secondaryBackgroundColor = '#1a1a1a'\n\
textColor = '#ffffff'\n\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
" > ~/.streamlit/config.toml
