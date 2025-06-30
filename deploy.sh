#!/bin/bash

# Cyberpunk Churn Prediction - Deployment Script
# This script handles deployment to various platforms

set -e

echo "🎮 CYBERPUNK CHURN PREDICTION - DEPLOYMENT SCRIPT"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker is installed"
}

# Check if Docker Compose is installed
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        print_warning "Docker Compose not found, trying docker compose..."
        if ! docker compose version &> /dev/null; then
            print_error "Docker Compose is not available. Please install Docker Compose."
            exit 1
        fi
        DOCKER_COMPOSE_CMD="docker compose"
    else
        DOCKER_COMPOSE_CMD="docker-compose"
    fi
    print_success "Docker Compose is available"
}

# Build Docker image
build_image() {
    print_status "Building Docker image..."
    docker build -t cyberpunk-churn-prediction:latest .
    print_success "Docker image built successfully"
}

# Deploy with Docker Compose
deploy_docker_compose() {
    print_status "Deploying with Docker Compose..."
    $DOCKER_COMPOSE_CMD down --remove-orphans
    $DOCKER_COMPOSE_CMD up -d --build
    print_success "Application deployed with Docker Compose"
    print_status "Application available at: http://localhost:8501"
}

# Deploy to Streamlit Cloud
deploy_streamlit_cloud() {
    print_status "Preparing for Streamlit Cloud deployment..."
    
    # Check if streamlit config exists
    if [ ! -f ".streamlit/config.toml" ]; then
        mkdir -p .streamlit
        cat > .streamlit/config.toml << EOF
[theme]
base = "dark"
primaryColor = "#00ffff"
backgroundColor = "#0d1117"
secondaryBackgroundColor = "#1a1a1a"
textColor = "#ffffff"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false
EOF
        print_success "Created Streamlit configuration"
    fi
    
    print_status "Streamlit Cloud deployment files ready"
    print_warning "Please push to GitHub and connect your repository to Streamlit Cloud"
    print_status "Visit: https://share.streamlit.io/"
}

# Deploy to Heroku
deploy_heroku() {
    print_status "Preparing for Heroku deployment..."
    
    # Create Procfile
    cat > Procfile << EOF
web: streamlit run streamlit_app_simple.py --server.port=\$PORT --server.address=0.0.0.0
EOF
    
    # Create runtime.txt
    cat > runtime.txt << EOF
python-3.11.9
EOF
    
    # Create setup.sh for Heroku
    cat > setup.sh << EOF
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
port = \$PORT\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
" > ~/.streamlit/config.toml
EOF
    
    chmod +x setup.sh
    
    print_success "Heroku deployment files created"
    print_warning "Run the following commands to deploy to Heroku:"
    echo "  heroku create your-app-name"
    echo "  git add ."
    echo "  git commit -m 'Deploy to Heroku'"
    echo "  git push heroku main"
}

# Deploy to Railway
deploy_railway() {
    print_status "Preparing for Railway deployment..."
    
    # Create railway.json
    cat > railway.json << EOF
{
  "\$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "startCommand": "streamlit run streamlit_app_simple.py --server.port=\$PORT --server.address=0.0.0.0",
    "healthcheckPath": "/_stcore/health",
    "healthcheckTimeout": 100,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
EOF
    
    print_success "Railway deployment configuration created"
    print_warning "Push to GitHub and connect your repository to Railway"
    print_status "Visit: https://railway.app/"
}

# Main deployment function
main() {
    echo "Select deployment option:"
    echo "1) Local Docker"
    echo "2) Docker Compose"
    echo "3) Streamlit Cloud"
    echo "4) Heroku"
    echo "5) Railway"
    echo "6) All deployment files"
    
    read -p "Enter your choice (1-6): " choice
    
    case $choice in
        1)
            check_docker
            build_image
            print_status "Running Docker container..."
            docker run -d -p 8501:8501 --name cyberpunk-churn cyberpunk-churn-prediction:latest
            print_success "Container started at http://localhost:8501"
            ;;
        2)
            check_docker
            check_docker_compose
            deploy_docker_compose
            ;;
        3)
            deploy_streamlit_cloud
            ;;
        4)
            deploy_heroku
            ;;
        5)
            deploy_railway
            ;;
        6)
            deploy_streamlit_cloud
            deploy_heroku
            deploy_railway
            print_success "All deployment files created"
            ;;
        *)
            print_error "Invalid choice. Please select 1-6."
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
