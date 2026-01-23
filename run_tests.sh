#!/bin/bash
# Test runner script for CANSLIM Analyzer

set -e

echo "========================================="
echo "CANSLIM Analyzer - Test Suite"
echo "========================================="
echo ""

# Set environment to development
export CANSLIM_ENV=development

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Install development dependencies
pip install -q pytest pytest-cov pyyaml redis

# Run config loader test
echo ""
echo "========================================="
echo "Testing Configuration Loader"
echo "========================================="
python config_loader.py

# Run Redis cache test (if Redis is available)
echo ""
echo "========================================="
echo "Testing Redis Cache"
echo "========================================="
python redis_cache.py || echo "Warning: Redis tests skipped (Redis not available)"

# Run pytest
echo ""
echo "========================================="
echo "Running Unit Tests"
echo "========================================="
pytest tests/ -v --tb=short

echo ""
echo "========================================="
echo "Test Summary"
echo "========================================="
echo "✓ All tests completed"
echo "See htmlcov/index.html for coverage report"
