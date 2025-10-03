# Contributing to Portfolio Optimization System

First off, thank you for considering contributing to this project! 🎉

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)

---

## 📜 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Our Standards

**Positive behaviors include:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community

**Unacceptable behaviors include:**
- Harassment or discriminatory language
- Trolling or inflammatory comments
- Personal or political attacks
- Publishing others' private information

---

## 🤝 How Can I Contribute?

### Reporting Bugs

**Before submitting a bug report:**
1. Check the [Issues](https://github.com/mohin-io/Mixed-Integer-Optimization-for-Portfolio-Selection/issues) page
2. Ensure you're using the latest version
3. Collect relevant information (error messages, logs, environment)

**How to submit a good bug report:**

```markdown
**Description**: Brief summary of the issue

**Steps to Reproduce**:
1. Run command `python demo.py`
2. Click on '...'
3. See error

**Expected Behavior**: What should happen

**Actual Behavior**: What actually happened

**Environment**:
- OS: Windows 11 / macOS 14 / Ubuntu 22.04
- Python Version: 3.10.13
- Package Versions: (paste `pip list`)

**Additional Context**: Screenshots, logs, etc.
```

### Suggesting Enhancements

**Enhancement suggestions are welcome!**

Areas we'd love help with:
- 🔧 New optimization algorithms (Simulated Annealing, Tabu Search)
- 📊 Additional portfolio strategies (Black-Litterman, Factor Models)
- 🌐 API endpoints (RESTful, GraphQL)
- 📈 Backtesting framework
- 🎨 UI/UX improvements
- 📚 Documentation enhancements

**Submit enhancement via:**
1. Open an Issue with label `enhancement`
2. Describe the feature and use case
3. Provide examples or mockups if applicable

### Your First Code Contribution

**Good first issues:**
- Look for issues labeled `good first issue`
- Documentation improvements
- Adding tests
- Fixing typos or formatting

**Getting started:**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 💻 Development Setup

### Prerequisites

- Python 3.10+
- Git
- Virtual environment tool

### Clone and Setup

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/Mixed-Integer-Optimization-for-Portfolio-Selection.git
cd Mixed-Integer-Optimization-for-Portfolio-Selection

# Add upstream remote
git remote add upstream https://github.com/mohin-io/Mixed-Integer-Optimization-for-Portfolio-Selection.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
```

### Code Quality Checks

```bash
# Format code with black
black src tests

# Lint with flake8
flake8 src tests --max-line-length=100

# Type checking with mypy
mypy src
```

---

## 🔀 Pull Request Process

### Before Submitting

1. **Update from upstream**
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

2. **Run tests**
   ```bash
   pytest tests/ -v
   ```

3. **Format code**
   ```bash
   black src tests
   ```

4. **Update documentation** if needed

### Submitting the PR

1. **Create a descriptive branch name**
   ```bash
   git checkout -b feature/add-simulated-annealing
   git checkout -b fix/correlation-matrix-bug
   git checkout -b docs/improve-quickstart
   ```

2. **Commit with clear messages**
   ```bash
   git add .
   git commit -m "feat: add simulated annealing optimizer

   - Implement SA algorithm with cooling schedule
   - Add tests for convergence
   - Update documentation

   Closes #42"
   ```

   **Commit message format:**
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation changes
   - `test:` adding tests
   - `refactor:` code refactoring
   - `chore:` maintenance tasks

3. **Push to your fork**
   ```bash
   git push origin feature/add-simulated-annealing
   ```

4. **Open Pull Request on GitHub**
   - Fill out the PR template
   - Link related issues
   - Request review

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring
- [ ] Performance improvement

## Checklist
- [ ] Tests pass locally
- [ ] Code follows style guidelines (black formatted)
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Added tests for new functionality

## Related Issues
Closes #XX
```

### Review Process

- Maintainer will review within 48 hours
- Address feedback promptly
- Be respectful and collaborative
- CI/CD must pass before merge

---

## 📏 Coding Standards

### Python Style Guide

**Follow PEP 8 with these specifics:**

```python
# Maximum line length: 100 characters
# Use black for automatic formatting

# Good
def calculate_portfolio_metrics(
    weights: pd.Series,
    returns: pd.DataFrame,
    risk_aversion: float = 2.5
) -> Dict[str, float]:
    """
    Calculate portfolio performance metrics.

    Args:
        weights: Portfolio weights (sum to 1)
        returns: Historical returns DataFrame
        risk_aversion: Risk aversion parameter

    Returns:
        Dictionary of metrics (return, volatility, sharpe)
    """
    pass
```

### Type Hints

**Always use type hints:**

```python
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np

def optimize(
    returns: pd.DataFrame,
    method: str = 'max_sharpe',
    constraints: Optional[Dict] = None
) -> pd.Series:
    """Function with type hints."""
    pass
```

### Docstrings

**Use Google-style docstrings:**

```python
def complex_function(param1: int, param2: str) -> List[float]:
    """
    One-line summary of function.

    More detailed description if needed. Explain the purpose,
    algorithm, and any important notes.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is negative
        TypeError: When param2 is not a string

    Example:
        >>> result = complex_function(42, "test")
        >>> print(result)
        [1.0, 2.0, 3.0]
    """
    pass
```

### Project Structure

```
src/
├── module_name/
│   ├── __init__.py
│   ├── core.py          # Core functionality
│   ├── utils.py         # Utility functions
│   └── types.py         # Type definitions

tests/
├── test_module_name.py  # Mirror src structure
```

---

## 🧪 Testing Guidelines

### Test Structure

```python
import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def sample_data():
    """Fixture for test data."""
    returns = pd.DataFrame(...)
    return returns

def test_function_basic(sample_data):
    """Test basic functionality."""
    result = my_function(sample_data)
    assert result is not None
    assert len(result) == len(sample_data)

def test_function_edge_case():
    """Test edge case: empty input."""
    with pytest.raises(ValueError):
        my_function(pd.DataFrame())

def test_function_reproducibility(sample_data):
    """Test that results are reproducible."""
    result1 = my_function(sample_data)
    result2 = my_function(sample_data)
    pd.testing.assert_series_equal(result1, result2)
```

### Test Coverage

- Aim for >80% test coverage
- Test all edge cases
- Test error handling
- Test with different data types

### Running Specific Tests

```bash
# Run single test file
pytest tests/test_optimization.py

# Run single test
pytest tests/test_optimization.py::test_mio_optimizer_basic

# Run with markers
pytest -m slow  # Run slow tests only
```

---

## 📝 Documentation

### README Updates

Update README.md when:
- Adding new features
- Changing API
- Adding examples

### Code Comments

```python
# Good comment: Explains WHY, not WHAT
# Use Ledoit-Wolf shrinkage to reduce estimation error in covariance matrix

# Bad comment: States the obvious
# Loop through assets
for asset in assets:
    ...
```

### Documentation Files

- `docs/PLAN.md` - Implementation roadmap
- `docs/ARCHITECTURE.md` - System design
- `docs/RESULTS.md` - Analysis results
- `docs/DEPLOYMENT.md` - Deployment guide

---

## 🎁 Recognition

Contributors will be:
- Listed in README.md
- Acknowledged in release notes
- Invited to project discussions

---

## 📞 Questions?

- **GitHub Issues**: For bugs and features
- **Discussions**: For questions and ideas
- **Email**: mohinhasin999@gmail.com

---

## 🙏 Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort!

---

**Happy Contributing! 🚀**
