# Copilot Coding Agent Instructions for M02-P0202

## Repository Overview

**Project Name:** SpamShield - Smart Spam Detector  
**Type:** Python Machine Learning Application with Streamlit UI  
**Size:** Small (~1,100 lines of Python code)  
**Primary Language:** Python 3.12+  
**Frameworks & Key Technologies:**
- Streamlit (web UI for ML demos)
- PyTorch 2.6.0 with CUDA 12.4 support
- scikit-learn 1.6.1 (classical ML models)
- transformers 4.54.1 (Hugging Face models)
- FAISS (Facebook AI Similarity Search)
- XGBoost, NLTK, imbalanced-learn

**Purpose:** This repository implements a spam message classification system using both traditional ML algorithms (Logistic Regression, SVM, Naive Bayes, Random Forest, XGBoost) and modern embedding-based approaches (FAISS with multilingual-e5-base). The Streamlit app provides data visualization, model comparison, and live prediction capabilities.

## Environment Setup

### Prerequisites
- **Python Version:** 3.12+ (project requires-python = ">=3.12")
- **Package Manager:** uv (https://docs.astral.sh/uv/)
- **IMPORTANT:** Install `uv` first via pip: `pip install uv`

### Initial Setup Steps (ALWAYS follow this order)

1. **Create Virtual Environment:**
   ```bash
   uv venv
   ```
   - Creates `.venv/` directory using Python 3.12.3
   - Activation message: "Activate with: source .venv/bin/activate"

2. **Activate Virtual Environment:**
   - Linux/macOS: `source .venv/bin/activate`
   - Windows: `.venv\Scripts\activate`
   - NOTE: For bash tool usage, activation is not necessary; reference `.venv/bin/python` directly

3. **Install Dependencies (CRITICAL - Takes ~30-60 seconds):**
   ```bash
   uv sync
   ```
   - Installs 128 packages including large PyTorch libraries (~732 MB download)
   - This command MUST complete successfully before any other operations
   - Creates/updates `uv.lock` file
   - Installs both production and dev dependencies (including ruff linter)

## Build & Validation Commands

### Linting (ALWAYS run before committing)

**Check Only:**
```bash
uvx ruff check
```
- Runs Ruff linter using ruff==0.12.4 or later from dev dependencies
- Uses config from pyproject.toml: line-length=100, target-version="py312"
- Common issues found: unused imports (F401), redefined variables (F811)
- Exit code 0 = no issues, Exit code 1 = issues found

**Auto-fix Issues:**
```bash
uvx ruff check --fix
```
- Automatically fixes common issues like unused imports
- Run `uvx ruff check` again after to confirm all issues resolved
- Most issues are auto-fixable with `--fix`

### Running the Application

**Via Command Line:**
```bash
streamlit run ./src/app.py
```
- Or use the venv directly: `.venv/bin/streamlit run ./src/app.py`
- Default port: 8501
- App includes 4 tabs: Data Overview, Model Comparison, FAISS, Live Prediction
- First run downloads NLTK data: punkt_tab, stopwords, averaged_perceptron_tagger_eng, wordnet

**Via VS Code:**
- Use launch configuration: "Debug Streamlit App (Windows - venv)"
- Located in `.vscode/launch.json`
- Runs on port 8501 with debugpy support

**Via Google Colab:**
- Notebook available: `notebooks/runbook.ipynb`
- Direct link provided in README.md

### Testing
**IMPORTANT:** This repository has NO formal test suite. Do not attempt to run pytest or unittest.
- Testing is done manually through the Streamlit UI
- Validation happens via the CI workflow (linting only)

## Project Structure

### Root Directory Layout
```
/home/runner/work/m02-p0202/m02-p0202/
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI configuration
├── .streamlit/
│   └── config.toml                   # Streamlit theme (base="light")
├── .vscode/
│   └── launch.json                   # VS Code debug configuration
├── data/
│   ├── spam.csv                      # Original dataset (1 header + 5,574 data rows)
│   └── spam_augmented.csv            # Augmented dataset (1 header + 9,625 data rows)
├── notebooks/
│   ├── faiss.ipynb                   # FAISS experimentation
│   ├── runbook.ipynb                 # Google Colab runbook
│   └── spam.ipynb                    # Main spam analysis notebook
├── src/
│   ├── __init__.py                   # Empty init file
│   ├── app.py                        # Main Streamlit application (502 lines)
│   ├── config.py                     # Configuration (SEED=42, TEST_SIZE=0.2)
│   ├── models.py                     # ML model training & FAISS (188 lines)
│   ├── utils.py                      # Data processing utilities (383 lines)
│   └── error_analysis.json           # FAISS error analysis results
├── .editorconfig                     # Editor configuration (spaces, LF, indent_size=4)
├── .gitignore                        # Includes .venv/, mlruns/, .ruff_cache/
├── pyproject.toml                    # Project dependencies & ruff config
├── uv.lock                           # Locked dependency versions
└── README.md                         # User documentation
```

### Key Source Files

**src/app.py** (main entry point):
- Streamlit UI with 4 tabs: Data Overview, Model Comparison, FAISS, Live Prediction
- Caches model loading with `@st.cache_resource` decorator
- Uses `intfloat/multilingual-e5-base` model for embeddings
- Model options from `StatisticalModelOptions` class

**src/config.py**:
- `AppConfig.SEED = 42` - Used for reproducibility
- `AppConfig.TEST_SIZE = 0.2` - Train/test split ratio
- `AppConfig.Path.RAW_DATA_FILE` - Points to data/spam.csv
- `AppConfig.Path.AUGMENTED_DATA_FILE` - Points to data/spam_augmented.csv
- `seed_everything()` function sets random seeds for reproducibility

**src/models.py**:
- `StatisticalModelOptions` class contains model names as constants
- `get_all_models()` returns default models (excludes Gaussian/Categorical NB)
- `train_model()` - Main training function
- `get_embeddings()` - Creates embeddings using transformers
- FAISS functions: `create_train_test_metadata()`, `evaluate_knn_accuracy()`
- All models use `random_state=AppConfig.SEED` for reproducibility

**src/utils.py**:
- `VectorizerOptions` class: TFIDF, BAG_OF_WORDS
- `preprocess_text()` - Text preprocessing pipeline (lowercase, remove punctuation, tokenize, remove stopwords, stem)
- `vectorize_tokenized_text()` - Converts preprocessed text to vectors
- `create_train_test_data()` - Splits data with optional SMOTE/ADASYN augmentation
- `load_data()` - Loads CSV with configurable column selection
- `plot_confusion_matrix()` - Returns matplotlib figure for Streamlit
- NLTK downloads: punkt_tab, stopwords, averaged_perceptron_tagger_eng, wordnet

### Configuration Files

**pyproject.toml:**
- Package name: "spamshield" (version 0.0.1)
- Python requirement: `>=3.12`
- PyTorch sources configured for CUDA 12.4 via pytorch-cu124 index
- Ruff config: line-length=100, target-version="py312"
- Dev dependencies: ruff>=0.12.4

**.editorconfig:**
- All files: indent_style=space, end_of_line=lf
- Python files: indent_size=4, insert_final_newline=true

## CI/CD Workflow

**GitHub Actions: .github/workflows/ci.yml**

Triggers:
- Push to main branch
- Pull requests (any target branch)

Job: `lint` (runs on ubuntu-latest)
1. Checkout code (actions/checkout@v4)
2. Set up Python 3.11 (NOTE: CI uses 3.11 but project specifies 3.12+ - this works because of backward compatibility)
3. Install uv (astral-sh/setup-uv@v3)
4. Create venv and sync: `uv venv && uv sync`
5. Run linting: `uvx ruff check`

**CRITICAL:** The CI will fail if `uvx ruff check` exits with non-zero code.

## Common Workflows & Important Notes

### Making Code Changes
1. Always run `uv sync` first if you haven't already in this session
2. Make your changes to Python files in `src/`
3. Run `uvx ruff check --fix` to auto-fix linting issues
4. Run `uvx ruff check` to verify no remaining issues
5. Test manually via Streamlit if changing app.py: `.venv/bin/streamlit run ./src/app.py`
6. Commit changes

### Adding Dependencies
- Add to pyproject.toml dependencies list
- Run `uv sync` to install and update uv.lock
- For dev dependencies, add to `[dependency-groups]` section
- PyTorch packages use special index (pytorch-cu124) - see existing config

### PyTorch & CUDA Notes
- PyTorch 2.6.0 configured for CUDA 12.4
- Source URLs point to https://download.pytorch.org/whl/cu124
- Works on both Linux and Windows (sys_platform markers)
- If you modify torch versions, maintain the custom index configuration

### Data Files
- **spam.csv:** Original dataset (1 header + 5,574 data rows), columns: Category, Message
- **spam_augmented.csv:** Back-translation augmented dataset (1 header + 9,625 data rows)
- Data loading handles: drop_duplicates, dropna, encoding issues
- Label encoding: "ham" -> 0, "spam" -> 1

### Model Training
- Models defined in `StatisticalModelOptions`: Logistic Regression, SVM, Multinomial NB, Random Forest, XGBoost
- All use `random_state=42` for reproducibility
- Training happens in-memory during Streamlit session
- No model persistence to disk by default

### FAISS Integration
- Uses `intfloat/multilingual-e5-base` for embeddings
- Cached with `@st.cache_resource` to avoid reloading
- K-NN with k values: 1, 3, 5
- IndexFlatIP (inner product) for similarity search

## Troubleshooting

### Issue: "command not found: uv"
**Solution:** Run `pip install uv` first

### Issue: Import errors or module not found
**Solution:** Run `uv sync` to install all dependencies

### Issue: Ruff check fails with unused imports
**Solution:** Run `uvx ruff check --fix` to auto-fix, then verify with `uvx ruff check`

### Issue: Streamlit app won't start
**Solutions:**
- Ensure dependencies installed: `uv sync`
- Use full path: `.venv/bin/streamlit run ./src/app.py`
- Check port 8501 not already in use

### Issue: NLTK data not found
**Solution:** The app automatically downloads NLTK data on first run (punkt_tab, stopwords, etc.)

### Issue: CI fails on Python version mismatch
**Note:** CI uses Python 3.11 but project requires 3.12+. This is acceptable due to backward compatibility. Local development should use 3.12+.

## Important Conventions

1. **Random State:** Always use `AppConfig.SEED` (42) for any random operations
2. **Test Size:** Use `AppConfig.TEST_SIZE` (0.2) for train/test splits
3. **Vectorizers:** Use constants from `VectorizerOptions` class, not string literals
4. **Model Names:** Use constants from `StatisticalModelOptions` class
5. **Imports:** Keep imports organized; ruff will flag unused imports
6. **Line Length:** Maximum 100 characters (enforced by ruff)
7. **Indentation:** 4 spaces for Python (enforced by .editorconfig)
8. **File Paths:** Use `AppConfig.Path` constants for data file references

## Trust These Instructions

These instructions have been validated by running actual commands and examining the codebase. When in doubt:
1. Follow the exact command sequences provided above
2. Check pyproject.toml for dependency information
3. Refer to existing code patterns in src/ files
4. Only search the codebase if information here is incomplete or contradictory

**Remember:** This is a small, focused project. Don't over-engineer solutions. Follow existing patterns, keep changes minimal, and always run linting before committing.
