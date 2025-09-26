# Contributing to Trading AI Dashboard

Thank you for your interest in contributing to Trading AI Dashboard! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

1. **Check existing issues** first to avoid duplicates
2. **Use the issue template** when creating new issues
3. **Provide detailed information** including:
   - Steps to reproduce the problem
   - Expected vs actual behavior
   - System information (OS, Python version, etc.)
   - Error messages and logs

### Suggesting Features

1. **Check the roadmap** to see if the feature is already planned
2. **Open a feature request** with detailed description
3. **Explain the use case** and why it would be valuable
4. **Consider implementation complexity** and maintenance burden

### Code Contributions

#### Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/Ethan-LWT/Trading-Helper.git
   cd Trading_AI
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

#### Development Guidelines

##### Code Style

- **Follow PEP 8** for Python code style
- **Use meaningful variable names** and function names
- **Add docstrings** to all functions and classes
- **Keep functions small** and focused on a single responsibility
- **Use type hints** where appropriate

##### Example:
```python
def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """
    Calculate Relative Strength Index (RSI) for given prices.
    
    Args:
        prices: List of closing prices
        period: RSI calculation period (default: 14)
        
    Returns:
        RSI value between 0 and 100
        
    Raises:
        ValueError: If prices list is too short or period is invalid
    """
    if len(prices) < period + 1:
        raise ValueError(f"Need at least {period + 1} prices for RSI calculation")
    
    # Implementation here...
    return rsi_value
```

##### File Organization

- **Keep related functionality together** in the same module
- **Use clear directory structure** following the existing pattern
- **Separate concerns** (data, strategy, UI, etc.)
- **Avoid circular imports**

##### Testing

- **Write tests** for new functionality
- **Test edge cases** and error conditions
- **Use meaningful test names** that describe what is being tested
- **Keep tests isolated** and independent

##### Documentation

- **Update README.md** if adding new features
- **Add inline comments** for complex logic
- **Document API changes** in docstrings
- **Update configuration examples** if needed

#### Commit Guidelines

##### Commit Message Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

##### Types
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

##### Examples
```bash
feat(strategy): add multi-timeframe RSI strategy

Add support for coordinated RSI signals across multiple timeframes
with configurable thresholds and position sizing.

Closes #123
```

```bash
fix(data): handle API rate limiting gracefully

Add exponential backoff retry logic for Alpha Vantage API calls
to prevent failures during high-frequency requests.

Fixes #456
```

#### Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Ensure all tests pass**
4. **Update CHANGELOG.md** with your changes
5. **Create a pull request** with:
   - Clear title and description
   - Reference to related issues
   - Screenshots for UI changes
   - Testing instructions

##### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for functionality
- [ ] Manual testing completed

## Screenshots (if applicable)

## Related Issues
Closes #123
```

## 🏗️ Development Setup

### Environment Setup

1. **Python 3.11+** is required
2. **Virtual environment** is recommended
3. **API keys** are needed for full functionality (see README.md)

### Local Development

1. **Start the development server**
   ```bash
   cd web
   python app.py
   ```

2. **Access the application**
   - Open `http://localhost:5000` in your browser
   - The server will auto-reload on code changes

### Testing

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_strategy.py

# Run with coverage
python -m pytest --cov=.
```

## 📋 Areas for Contribution

### High Priority
- **Bug fixes** and stability improvements
- **Performance optimizations**
- **Test coverage** improvements
- **Documentation** enhancements

### Medium Priority
- **New technical indicators**
- **Additional data sources**
- **UI/UX improvements**
- **Mobile responsiveness**

### Low Priority
- **New AI models** integration
- **Advanced charting** features
- **Social features**
- **Cryptocurrency** support

## 🔍 Code Review Process

### For Contributors
- **Be responsive** to feedback
- **Make requested changes** promptly
- **Ask questions** if feedback is unclear
- **Test thoroughly** before requesting review

### For Reviewers
- **Be constructive** and helpful
- **Focus on code quality** and maintainability
- **Check for security issues**
- **Verify functionality** works as expected

## 🚀 Release Process

1. **Version bumping** follows semantic versioning
2. **Changelog** is updated for each release
3. **Testing** is performed on staging environment
4. **Documentation** is updated as needed

## 📞 Getting Help

- **GitHub Discussions** for general questions
- **GitHub Issues** for bug reports
- **Code comments** for implementation questions
- **Documentation** for usage questions

## 🏆 Recognition

Contributors will be recognized in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **GitHub contributors** page

## 📄 License

By contributing to Trading AI Dashboard, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Trading AI Dashboard! 🚀