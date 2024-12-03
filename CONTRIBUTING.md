# Contributing to CodeInsight

First off, thank you for considering contributing to CodeInsight! It's people like you that make CodeInsight such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include code samples and stack traces if applicable

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* Use a clear and descriptive title
* Provide a step-by-step description of the suggested enhancement
* Provide specific examples to demonstrate the steps
* Describe the current behavior and explain which behavior you expected to see instead
* Explain why this enhancement would be useful
* List some other tools or applications where this enhancement exists

### Pull Requests

* Fill in the required template
* Do not include issue numbers in the PR title
* Include screenshots and animated GIFs in your pull request whenever possible
* Follow the Python style guide
* Include thoughtfully-worded, well-structured tests
* Document new code based on the Documentation Styleguide
* End all files with a newline

## Development Process

1. Fork the repo
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Setup Development Environment

```bash
# Clone your fork
git clone https://github.com/your-username/codeinsight.git
cd codeinsight

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_security.py

# Run with coverage
pytest --cov=codeinsight tests/
```

### Code Style

We use `black` for code formatting and `flake8` for style guide enforcement:

```bash
# Format code
black .

# Check style
flake8
```

## Documentation Styleguide

* Use [Google style](https://google.github.io/styleguide/pyguide.html) for docstrings
* Use [Markdown](https://guides.github.com/features/mastering-markdown/) for documentation

Example:
```python
def analyze_code(code: str, language: str) -> Dict[str, Any]:
    """Analyzes code for patterns and issues.
    
    Args:
        code: The source code to analyze.
        language: The programming language of the code.
        
    Returns:
        A dictionary containing analysis results with keys:
        - 'patterns': List of detected patterns
        - 'issues': List of potential issues
        
    Raises:
        ValueError: If language is not supported.
    """
```

## AI Model Development

When contributing to AI models:

1. Document model architecture and training process
2. Include model cards following [Hugging Face guidelines](https://huggingface.co/docs/hub/model-cards)
3. Provide training and evaluation scripts
4. Include performance benchmarks
5. Document model limitations and biases

## Testing Guidelines

* Write unit tests for all new code
* Maintain test coverage above 90%
* Mock external dependencies
* Use meaningful test names that describe the behavior being tested
* Use fixtures for common test setup
* Include both positive and negative test cases

## Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

## Additional Notes

### Issue and Pull Request Labels

* `bug`: Something isn't working
* `enhancement`: New feature or request
* `documentation`: Improvements or additions to documentation
* `good first issue`: Good for newcomers
* `help wanted`: Extra attention is needed
* `performance`: Performance-related improvements
* `security`: Security-related improvements
* `testing`: Testing-related tasks

## Recognition

Contributors who make significant improvements will be recognized in our README and release notes.

Thank you for contributing to CodeInsight! 🎉
