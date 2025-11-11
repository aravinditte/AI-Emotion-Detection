# Contributing to AI Emotion Detection

Thank you for your interest in contributing to this project!

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists
2. Create a new issue with:
   - Clear description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (OS, Python version)
   - Screenshots if applicable

### Code Contributions

#### Getting Started

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub, then:
   git clone https://github.com/YOUR_USERNAME/AI-Emotion-Detection.git
   cd AI-Emotion-Detection
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set up development environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements_full.txt
   ```

#### Development Guidelines

##### Code Style

- Follow PEP 8 style guide
- Use type hints for function parameters and returns
- Write descriptive variable names
- Add docstrings to all classes and functions
- Keep functions focused and small

##### Example Function

```python
def process_frame(frame: np.ndarray, 
                  threshold: float = 0.5) -> Tuple[bool, str]:
    """Process a video frame for emotion detection.
    
    Args:
        frame: Input BGR image from webcam
        threshold: Confidence threshold (0.0 to 1.0)
        
    Returns:
        Tuple of (success, emotion_label)
    """
    # Implementation
    pass
```

##### Testing

- Test your changes thoroughly
- Ensure existing functionality still works
- Test on different systems if possible
- Add test cases for new features

##### Documentation

- Update README.md if adding features
- Update SETUP.md if changing installation
- Add inline comments for complex logic
- Update docstrings when modifying functions

#### Making Changes

1. **Write your code**
   - Follow the guidelines above
   - Commit regularly with clear messages
   - Keep commits focused and atomic

2. **Commit messages**
   ```bash
   # Good commit messages:
   git commit -m "Add support for custom color themes"
   git commit -m "Fix camera initialization bug on macOS"
   git commit -m "Improve emotion detection accuracy"
   
   # Avoid:
   git commit -m "Update stuff"
   git commit -m "Fix bug"
   ```

3. **Push your changes**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create Pull Request**
   - Go to GitHub and create a pull request
   - Provide clear description of changes
   - Reference any related issues
   - Wait for review

#### Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No unnecessary files included
- [ ] Code is commented where needed

### Areas for Contribution

#### Easy (Good First Issues)
- Fix typos in documentation
- Add more persona mappings
- Improve error messages
- Add keyboard shortcut documentation

#### Medium
- Add new visual themes
- Improve heuristic emotion detection
- Add unit tests
- Optimize rendering performance

#### Advanced
- Multi-face tracking support
- Web interface implementation
- Mobile app development
- Real-time emotion graphing
- Custom model training

### Code Review Process

1. Maintainer reviews your PR
2. Feedback provided if changes needed
3. You make requested changes
4. Once approved, PR is merged

### Community Guidelines

- Be respectful and professional
- Provide constructive feedback
- Help others when possible
- Follow the code of conduct

### Questions?

If you have questions:
- Open an issue with the "question" label
- Check existing issues and documentation first
- Be specific about what you need help with

## Development Setup Tips

### Virtual Environment

Always use a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

### IDE Setup

**VS Code**:
- Install Python extension
- Enable type checking
- Use pylint or flake8

**PyCharm**:
- Configure Python interpreter
- Enable code inspections
- Use built-in debugger

### Debugging

For debugging visual issues:
```python
# Add debug visualization
cv2.imshow('Debug', debug_frame)
cv2.waitKey(0)
```

For debugging emotion detection:
```python
# Print intermediate values
print(f"Mouth width: {mouth_width}, Height: {mouth_height}")
print(f"Eye aspect: {eye_aspect}")
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to AI Emotion Detection!
