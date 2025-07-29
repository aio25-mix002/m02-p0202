# M02-P0202 Project

- [M02-P0202 Project](#m02-p0202-project)
  - [Prerequisites](#prerequisites)
  - [Development Setup](#development-setup)
    - [Step 1: Create the virtual environment](#step-1-create-the-virtual-environment)
    - [Step 2: Activate the virtual environment](#step-2-activate-the-virtual-environment)
    - [Step 3: Install dependencies](#step-3-install-dependencies)
  - [Development Guidelines](#development-guidelines)
    - [Code Linting](#code-linting)
    - [CI/CD](#cicd)


## Prerequisites
- Python: python 3.11 
- Package Management: uv

## Development Setup

### Step 1: Create the virtual environment
```
uv venv
```

### Step 2: Activate the virtual environment

Windows: 
```
.venv\Scripts\activate
```

### Step 3: Install dependencies
```
uv sync
```

## Development Guidelines

### Code Linting
```
uvx ruff check
```

### CI/CD
The project includes a GitHub Actions workflow that automatically runs linting checks on:
- Pull requests to main branch
- Pushes to main branch
