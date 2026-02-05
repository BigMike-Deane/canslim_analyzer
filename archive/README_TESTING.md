# Testing Guide for CANSLIM Analyzer

## Quick Start

```bash
# Run all tests
export CANSLIM_ENV=development
python3 -m pytest tests/ -v

# Run with coverage
python3 -m pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html

# Run specific test file
python3 -m pytest tests/test_canslim_scorer.py -v

# Run specific test
python3 -m pytest tests/test_canslim_scorer.py::TestCANSLIMScorer::test_full_canslim_score -v
```

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── test_canslim_scorer.py         # Scoring logic tests (19 tests)
├── test_config_loader.py          # Configuration tests (10 tests)
└── test_redis_cache.py            # Redis cache tests (8 tests)
```

## Running Tests with Redis

### Option 1: Docker (Recommended)
```bash
# Start Redis
docker run -d -p 6379:6379 --name test-redis redis:7-alpine

# Run tests
python3 -m pytest tests/ -v

# Stop Redis
docker stop test-redis && docker rm test-redis
```

### Option 2: System Redis
```bash
# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis

# macOS
brew install redis
brew services start redis

# Run tests
python3 -m pytest tests/ -v
```

## Test Coverage

Current coverage (as of 2026-01-23):
- Config Loader: 95.59%
- Redis Cache: 32.19%
- CANSLIM Scorer: 47.37%
- Overall: 11.73% (focused on new modules)

Target: 80%+ for critical modules

## Writing New Tests

### Add test to existing file:
```python
# tests/test_canslim_scorer.py

def test_new_scoring_feature(self, mock_stock_data, mock_data_fetcher):
    """Test description"""
    scorer = CANSLIMScorer(mock_data_fetcher)
    
    # Arrange
    mock_stock_data.some_value = 123
    
    # Act
    score, detail = scorer._score_something(mock_stock_data)
    
    # Assert
    assert score > 0, "Should have positive score"
    assert "expected" in detail
```

### Create new test file:
```python
# tests/test_new_module.py

import pytest
from new_module import NewClass

class TestNewClass:
    """Test new functionality"""
    
    def test_something(self):
        obj = NewClass()
        result = obj.do_something()
        assert result == expected
```

## Continuous Integration

Add to your CI/CD pipeline:
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: pytest tests/ --cov=. --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Debugging Tests

### Run with verbose output:
```bash
python3 -m pytest tests/ -vv
```

### Run with print statements:
```bash
python3 -m pytest tests/ -s
```

### Run only failed tests:
```bash
python3 -m pytest tests/ --lf
```

### Drop into debugger on failure:
```bash
python3 -m pytest tests/ --pdb
```

## Test Markers

Mark tests for selective running:
```python
@pytest.mark.slow
def test_full_scan():
    # Long-running test
    pass

@pytest.mark.redis
def test_redis_feature():
    # Requires Redis
    pass
```

Run marked tests:
```bash
# Only slow tests
python3 -m pytest -m slow

# Skip slow tests
python3 -m pytest -m "not slow"

# Only Redis tests (if Redis available)
python3 -m pytest -m redis
```

## Common Issues

### "ModuleNotFoundError"
```bash
# Add project to PYTHONPATH
export PYTHONPATH=/path/to/canslim_analyzer:$PYTHONPATH
```

### Redis connection refused
```bash
# Redis tests will skip automatically
# Or start Redis:
docker run -d -p 6379:6379 redis:7-alpine
```

### Import errors
```bash
# Ensure dependencies installed
pip install -r requirements.txt
pip install pytest pytest-cov pyyaml redis
```
