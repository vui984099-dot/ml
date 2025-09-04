#!/bin/bash

# Amazon Product Q&A System Startup Script

echo "🚀 Amazon Product Q&A & Recommendation System"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

print_status "Python 3 found"

# Check if pip is available
if ! command -v pip &> /dev/null; then
    print_error "pip is required but not installed"
    exit 1
fi

print_status "pip found"

# Install dependencies
print_info "Installing Python dependencies..."
if pip install -r requirements.txt > /dev/null 2>&1; then
    print_status "Dependencies installed successfully"
else
    print_warning "Some dependencies may have failed to install"
fi

# Create necessary directories
print_info "Creating directories..."
mkdir -p data/raw data/parquet data/indexes models logs
print_status "Directories created"

# Initialize database and run data pipeline
print_info "Running data pipeline..."
if python run_demo.py > logs/setup.log 2>&1; then
    print_status "Data pipeline completed successfully"
else
    print_warning "Data pipeline had issues, check logs/setup.log"
fi

# Check what services to start
echo ""
echo "Choose how to start the system:"
echo "1) Docker (recommended)"
echo "2) Manual (separate terminals)"
echo "3) Background processes"
echo "4) Just show instructions"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        print_info "Starting with Docker..."
        if command -v docker-compose &> /dev/null; then
            docker-compose up --build
        else
            print_error "docker-compose not found. Please install Docker and docker-compose"
            exit 1
        fi
        ;;
    2)
        echo ""
        print_info "Manual startup instructions:"
        echo ""
        echo "Terminal 1 (API Backend):"
        echo "  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload"
        echo ""
        echo "Terminal 2 (Streamlit UI):"
        echo "  streamlit run src/ui/app.py --server.port 8501"
        echo ""
        print_info "After starting both services:"
        echo "  • API: http://localhost:8000"
        echo "  • UI: http://localhost:8501"
        echo "  • Docs: http://localhost:8000/docs"
        ;;
    3)
        print_info "Starting services in background..."
        
        # Start API in background
        nohup uvicorn src.api.main:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &
        API_PID=$!
        echo $API_PID > logs/api.pid
        print_status "API started (PID: $API_PID)"
        
        # Wait a moment for API to start
        sleep 5
        
        # Start UI in background
        nohup streamlit run src/ui/app.py --server.port 8501 > logs/ui.log 2>&1 &
        UI_PID=$!
        echo $UI_PID > logs/ui.pid
        print_status "UI started (PID: $UI_PID)"
        
        echo ""
        print_status "Services running in background!"
        echo "  • API: http://localhost:8000 (PID: $API_PID)"
        echo "  • UI: http://localhost:8501 (PID: $UI_PID)"
        echo "  • Logs: logs/api.log, logs/ui.log"
        echo ""
        echo "To stop services:"
        echo "  kill $API_PID $UI_PID"
        echo "  # or run: pkill -f 'uvicorn\\|streamlit'"
        ;;
    4)
        print_info "System is ready! Here's how to start:"
        echo ""
        echo "🐳 Docker (Recommended):"
        echo "   docker-compose up"
        echo ""
        echo "🔧 Manual:"
        echo "   make api    # Start API backend"
        echo "   make ui     # Start UI frontend (separate terminal)"
        echo ""
        echo "📋 Available commands:"
        echo "   make help   # Show all available commands"
        echo "   make test   # Run test suite"
        echo "   make status # Check system status"
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

echo ""
print_status "Setup completed! 🎉"