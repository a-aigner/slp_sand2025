# Audio Classification ML Project - Complete Guide

A command-line application for training machine learning models on audio (.wav) files. This tool extracts features from audio and uses them to predict classifications.

## Table of Contents

1. [What This Does](#what-this-does)
2. [Complete Setup Instructions](#complete-setup-instructions)
3. [Project Structure](#project-structure)
4. [Quick Start Examples](#quick-start-examples)
5. [All Command-Line Parameters](#all-command-line-parameters)
6. [Training Modes](#training-modes)
7. [Understanding Features](#understanding-features)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Usage](#advanced-usage)

---

## What This Does

This tool helps you:
- Train machine learning models to classify audio recordings
- Extract audio features automatically (MFCCs, spectral features, etc.)
- Predict categories from audio files
- Combine audio with demographic data (Age, Sex) to predict outcomes (Class 1-5)
- Evaluate model performance with accuracy scores and confusion matrices

**Example Use Cases:**
- Classify different types of phonation (vowel sounds)
- Predict health outcomes from voice recordings
- Categorize audio by rhythm patterns
- Combine voice features with patient data for medical predictions

---

## Complete Setup Instructions

### Prerequisites

You need Python installed on your computer. This works with Python 3.8 or newer.

**Check if you have Python:**
```bash
python --version
# or
python3 --version
```

If you don't have Python, download it from [python.org](https://www.python.org/downloads/)

### Step 1: Open Terminal/Command Prompt

**On macOS:**
- Press `Cmd + Space`, type "Terminal", press Enter

**On Windows:**
- Press `Win + R`, type "cmd", press Enter

**On Linux:**
- Press `Ctrl + Alt + T`

### Step 2: Navigate to Project Directory

```bash
# Replace with your actual path
cd /Users/andreaigner/Dev/projects/pycharm/SLP/setup
```

### Step 3: Create a Virtual Environment

A virtual environment keeps this project's dependencies separate from other Python projects.

**On macOS/Linux:**
```bash
python3 -m venv venv
```

**On Windows:**
```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

You'll see `(venv)` appear at the start of your command line when it's active.

### Step 5: Install Required Packages

```bash
pip install -r requirements.txt
```

This installs all the necessary libraries:
- `librosa` - Audio feature extraction
- `scikit-learn` - Machine learning algorithms
- `pandas` - Data handling
- `numpy` - Numerical operations
- `matplotlib` & `seaborn` - Visualizations
- `openpyxl` - Excel file reading

**Installation takes 2-5 minutes.** Wait for it to complete.

### Step 6: Verify Installation

```bash
python train.py --help
```

If you see a help message with all the options, you're ready to go!

---

## Project Structure

```
setup/
├── data/                          # Your audio data
│   ├── task1/
│   │   ├── sand_task_1.xlsx       # Metadata (ID, Age, Sex, Class)
│   │   └── training/
│   │       ├── phonationA/        # Vowel 'A' recordings (.wav files)
│   │       ├── phonationE/        # Vowel 'E' recordings
│   │       ├── phonationI/        # Vowel 'I' recordings
│   │       ├── phonationO/        # Vowel 'O' recordings
│   │       ├── phonationU/        # Vowel 'U' recordings
│   │       ├── rhythmKA/          # 'KA' rhythm recordings
│   │       ├── rhythmPA/          # 'PA' rhythm recordings
│   │       └── rhythmTA/          # 'TA' rhythm recordings
│   └── task2/
│       ├── sand_task_2.xlsx
│       └── training/
│
├── src/                           # Python modules (don't edit unless you know what you're doing)
│   ├── feature_extraction.py     # Extracts features from audio
│   ├── temporal_features.py      # Frame-based feature extraction
│   ├── data_loader.py            # Loads audio files
│   ├── metadata_loader.py        # Loads Excel metadata
│   └── trainer.py                # Trains ML models
│
├── models/                        # Saved models appear here (created automatically)
│   └── plots/                    # Confusion matrices and plots
│
├── train.py                       # MAIN SCRIPT - Run this to train models
├── predict.py                     # Use trained models for predictions
├── requirements.txt               # List of required Python packages
└── README.md                      # This file
```

---

## Quick Start Examples

### Example 1: Simplest Training (2 Classes)

Train a model to distinguish between two types of audio:

```bash
python train.py \
    --data-dir data/task1/training \
    --subdirs phonationA phonationE
```

**What this does:**
- Loads .wav files from `phonationA` and `phonationE` folders
- Extracts audio features automatically
- Trains a Random Forest classifier (default)
- Uses 80% data for training, 20% for testing
- Prints accuracy results

### Example 2: Using Metadata to Predict Class

Use audio features + Age + Sex to predict Class (1-5):

```bash
python train.py \
    --data-dir data/task1/training \
    --use-metadata \
    --excel-file data/task1/sand_task_1.xlsx \
    --model-type logistic_regression
```

**What this does:**
- Loads all .wav files from all subdirectories
- Matches each file with Excel data by ID
- Combines audio features with Age and Sex
- Predicts Class values (1-5) from Excel
- Uses Logistic Regression model

### Example 3: Temporal Features (Frame-Based)

Extract features from short time windows (like speech processing):

```bash
python train.py \
    --data-dir data/task1/training \
    --subdirs phonationA phonationE phonationI \
    --use-temporal \
    --frame-length 20.0 \
    --save-plots
```

**What this does:**
- Processes audio in 20ms windows
- Extracts features every 10ms
- Includes velocity (delta) and acceleration features
- Saves confusion matrix plot

### Example 4: Everything Combined

```bash
python train.py \
    --data-dir data/task1/training \
    --use-temporal \
    --frame-length 20.0 \
    --context-frames 1 \
    --use-metadata \
    --excel-file data/task1/sand_task_1.xlsx \
    --model-type random_forest \
    --save-plots \
    --model-name complete_model
```

**What this does:**
- Temporal features with 20ms frames
- Stacks 3 consecutive frames for context
- Combines with Age/Sex metadata
- Predicts Class (1-5)
- Saves everything with custom name

---

## All Command-Line Parameters

### How to Use Parameters

Parameters are added after `python train.py` like this:

```bash
python train.py --parameter-name value --another-parameter value
```

**Boolean flags** (on/off) don't need a value:
```bash
python train.py --use-temporal --save-plots
```

### Data Parameters

#### `--data-dir` (REQUIRED)
**What it does:** Specifies where your audio files are located

**Type:** Path to a directory

**Examples:**
```bash
--data-dir data/task1/training
--data-dir /full/path/to/audio/files
```

**When to use:** Always required. Point to the folder containing your audio subdirectories.

---

#### `--subdirs`
**What it does:** Choose which subdirectories to use for training

**Type:** List of folder names (space-separated)

**Default:** Uses all subdirectories if not specified

**Examples:**
```bash
--subdirs phonationA phonationE
--subdirs rhythmKA rhythmPA rhythmTA
--subdirs phonationA
```

**When to use:** 
- When you only want to train on specific test types
- To compare performance across different audio types
- To exclude certain folders

**Notes:**
- Folder names must be spelled exactly as they appear
- Separate multiple folders with spaces
- Can use just one folder

---

#### `--label`
**What it does:** Assign a single label to all files in one directory

**Type:** String

**Default:** None

**Example:**
```bash
python train.py --data-dir data/task1/training/phonationA --label phonationA
```

**When to use:** 
- Single-directory mode (rarely needed)
- Cannot be used with `--use-metadata`

**Note:** Most users should use `--subdirs` instead

---

#### `--use-metadata`
**What it does:** Enable metadata mode - combines audio with Age/Sex to predict Class

**Type:** Boolean flag (no value needed)

**Default:** Off

**Example:**
```bash
--use-metadata
```

**When to use:**
- When you want to use Age and Sex as additional features
- When predicting Class values (1-5) from Excel file
- Must be used with `--excel-file`

**What happens:**
- Reads ID, Age, Sex, Class from Excel
- Matches .wav files to Excel by ID in filename
- Uses Age + Sex as features
- Uses Class as prediction target (not folder names)

---

#### `--excel-file`
**What it does:** Path to Excel file containing metadata

**Type:** Path to .xlsx file

**Default:** None

**Example:**
```bash
--excel-file data/task1/sand_task_1.xlsx
```

**When to use:** Required when using `--use-metadata`

**Excel file must have these columns:**
- `ID` - Identifier that appears in .wav filename
- `Age` - Numeric age value
- `Sex` - Gender (M/F, Male/Female, or 0/1)
- `Class` - Target class to predict (1-5 or any integers)

---

### Model Parameters

#### `--model-type`
**What it does:** Choose which machine learning algorithm to use

**Type:** Choice from list

**Options:**
- `random_forest` (default) - Ensemble of decision trees, robust, good default
- `svm` - Support Vector Machine, often high accuracy, slower
- `logistic_regression` - Fast, interpretable, good for multi-class
- `linear_regression` - Predicts continuous values, rounds to nearest class

**Default:** `random_forest`

**Examples:**
```bash
--model-type random_forest
--model-type logistic_regression
--model-type svm
--model-type linear_regression
```

**When to use each:**

**Random Forest:**
- Good default choice
- Handles non-linear patterns well
- Shows feature importance
- Robust to outliers
```bash
--model-type random_forest
```

**Logistic Regression:**
- Fast training
- Works well for class prediction (1-5)
- Interpretable results
- Good with many features
```bash
--model-type logistic_regression
```

**SVM:**
- Often highest accuracy
- Good for complex patterns
- Slower with large datasets
```bash
--model-type svm
```

**Linear Regression:**
- Predicts numeric Class values directly
- Simple and fast
- Rounds predictions to nearest integer
```bash
--model-type linear_regression
```

---

#### `--test-size`
**What it does:** Fraction of data used for testing (validation)

**Type:** Decimal between 0 and 1

**Default:** `0.2` (20% test, 80% train)

**Examples:**
```bash
--test-size 0.2    # 20% test, 80% train (default)
--test-size 0.3    # 30% test, 70% train
--test-size 0.1    # 10% test, 90% train
```

**When to use:**
- `0.2` (default) - Standard for most datasets
- `0.3` - When you have lots of data
- `0.1` - When you have limited data

**How it works:**
- Training data: Used to learn patterns
- Test data: Used to evaluate performance (unseen data)
- Split is random but reproducible

---

### Audio Processing Parameters

#### `--sample-rate`
**What it does:** Audio sample rate for loading .wav files

**Type:** Integer (Hz)

**Default:** `22050` (22.05 kHz)

**Examples:**
```bash
--sample-rate 22050    # Default, good for speech
--sample-rate 16000    # Faster processing
--sample-rate 44100    # CD quality (slower)
```

**When to use:**
- `22050` (default) - Standard for speech analysis
- `16000` - Faster, still good for voice
- `44100` - High quality, slower processing

**Note:** Lower sample rate = faster processing but less audio detail

---

#### `--n-mfcc`
**What it does:** Number of MFCC (Mel-frequency cepstral coefficients) to extract

**Type:** Integer

**Default:** `13`

**Examples:**
```bash
--n-mfcc 13    # Default, standard in speech processing
--n-mfcc 20    # More detail
--n-mfcc 40    # Maximum detail
```

**When to use:**
- `13` (default) - Standard, works for most cases
- `20` - Try if you want more detail
- `40` - Maximum detail (more features but slower)

**What are MFCCs?** Features that represent the shape of the audio spectrum, like a "fingerprint" of the sound.

---

### Temporal Feature Parameters

#### `--use-temporal`
**What it does:** Enable frame-based temporal feature extraction

**Type:** Boolean flag

**Default:** Off

**Example:**
```bash
--use-temporal
```

**When to use:**
- When you want to capture how audio changes over time
- For speech/voice analysis
- When you need velocity and acceleration features

**What it does:**
- Splits audio into short frames (windows)
- Extracts features from each frame
- Includes delta (velocity) features
- Includes delta-delta (acceleration) features
- Aggregates across time for final features

**Without temporal:** One feature vector per entire audio file (66 features)
**With temporal:** Features from many time frames, aggregated (78+ features)

---

#### `--frame-length`
**What it does:** Length of each frame/window in milliseconds

**Type:** Decimal (milliseconds)

**Default:** `25.0` ms

**Examples:**
```bash
--frame-length 25.0    # Default, standard speech processing
--frame-length 20.0    # Recommended for detailed speech analysis
--frame-length 15.0    # More time resolution
--frame-length 10.0    # Maximum detail
```

**When to use:**
- `25.0` (default) - Standard, works well
- `20.0` - Recommended for speech
- `15.0` - More detail, slower
- `10.0` - Maximum detail, slowest

**Recommended range:** 10-25 ms

**How it works:** Audio is split into overlapping windows of this length

---

#### `--hop-length`
**What it does:** Step size between frames in milliseconds

**Type:** Decimal (milliseconds)

**Default:** `10.0` ms

**Examples:**
```bash
--hop-length 10.0    # Default, standard overlap
--hop-length 5.0     # More overlap, more frames
--hop-length 15.0    # Less overlap, fewer frames
```

**When to use:**
- `10.0` (default) - Standard, good overlap
- `5.0` - Maximum overlap, more computation
- `15.0` - Less overlap, faster

**Rule of thumb:** Hop length should be 40-50% of frame length

**Example:** With frame=25ms and hop=10ms:
- Frame 1: 0-25ms
- Frame 2: 10-35ms (15ms overlap with Frame 1)
- Frame 3: 20-45ms (15ms overlap with Frame 2)

---

#### `--use-deltas`
**What it does:** Include delta (velocity) features

**Type:** Boolean flag

**Default:** On (True)

**Examples:**
```bash
# Enable (default)
--use-deltas

# Disable
--no-use-deltas
```

**When to use:**
- Keep enabled (default) for most cases
- Disable only if you want fewer features

**What are deltas?** 
- First derivative of features
- Shows how fast features are changing
- Captures dynamics like vowel transitions

**Impact on features:**
- Without: 13 MFCCs per frame
- With: 13 MFCCs + 13 deltas = 26 features per frame

---

#### `--use-delta-deltas`
**What it does:** Include delta-delta (acceleration) features

**Type:** Boolean flag

**Default:** On (True)

**Examples:**
```bash
# Enable (default)
--use-delta-deltas

# Disable
--no-use-delta-deltas
```

**When to use:**
- Keep enabled (default) for most cases
- Disable if you want fewer features or faster processing

**What are delta-deltas?**
- Second derivative of features
- Shows how the rate of change is changing
- Captures acceleration in audio dynamics

**Impact on features:**
- Without: 26 features per frame (MFCCs + deltas)
- With: 39 features per frame (MFCCs + deltas + delta-deltas)

---

#### `--context-frames`
**What it does:** Number of frames before/after to stack together

**Type:** Integer (0 or higher)

**Default:** `0` (no stacking)

**Examples:**
```bash
--context-frames 0    # No stacking (default)
--context-frames 1    # Stack 3 frames: [t-1, t, t+1]
--context-frames 2    # Stack 5 frames: [t-2, t-1, t, t+1, t+2]
--context-frames 3    # Stack 7 frames: [t-3, t-2, t-1, t, t+1, t+2, t+3]
```

**When to use:**
- `0` (default) - Start here, usually sufficient
- `1` - Try if you need temporal context
- `2-3` - For complex temporal patterns

**What it does:** Combines consecutive frames to capture relationships between neighboring time windows

**Impact on features:**
- `context-frames=0`: 39 features per frame
- `context-frames=1`: 117 features per frame (39 × 3)
- `context-frames=2`: 195 features per frame (39 × 5)

**Trade-off:** More context = more information but also more features (need more data)

---

### Output Parameters

#### `--model-dir`
**What it does:** Directory where trained models are saved

**Type:** Path to directory

**Default:** `models`

**Examples:**
```bash
--model-dir models              # Default
--model-dir saved_models
--model-dir /full/path/to/save
```

**When to use:** If you want to organize models in different folders

**Note:** Directory is created automatically if it doesn't exist

---

#### `--model-name`
**What it does:** Custom name for the saved model

**Type:** String

**Default:** Auto-generated with timestamp (e.g., `random_forest_20251009_143022`)

**Examples:**
```bash
--model-name my_phonation_model
--model-name experiment_1
--model-name phonation_rf_v2
```

**When to use:**
- To organize multiple experiments
- To give models descriptive names
- To avoid timestamp names

**Note:** Model file will be saved as `{model-name}.pkl`

---

#### `--save-plots`
**What it does:** Save confusion matrix and feature importance plots

**Type:** Boolean flag

**Default:** Off

**Example:**
```bash
--save-plots
```

**When to use:**
- Always recommended for analyzing results
- To see which classes are confused
- To identify important features (Random Forest only)

**What gets saved:**
- `models/plots/confusion_matrix_{model_type}.png`
- `models/plots/feature_importance_{model_type}.png` (Random Forest only)

---

#### `--random-seed`
**What it does:** Random seed for reproducibility

**Type:** Integer

**Default:** `42`

**Examples:**
```bash
--random-seed 42     # Default
--random-seed 123
--random-seed 2024
```

**When to use:**
- Keep default for consistent results
- Change to get different random splits
- Use same seed to reproduce exact results

**What it affects:**
- Train/test split
- Random initialization in models

---

## Training Modes

### Mode 1: Standard Classification (Folder-Based)

**Goal:** Classify audio by folder names

**Command structure:**
```bash
python train.py --data-dir DIRECTORY --subdirs FOLDER1 FOLDER2 ...
```

**How it works:**
- Each subdirectory represents a class
- Folder name = class label
- All .wav files in folder get that label

**Example:**
```bash
python train.py \
    --data-dir data/task1/training \
    --subdirs phonationA phonationE phonationI
```

**Result:** Model learns to distinguish between phonationA, phonationE, and phonationI

---

### Mode 2: Metadata-Based Classification

**Goal:** Predict Class (1-5) using audio + Age + Sex

**Command structure:**
```bash
python train.py --data-dir DIRECTORY --use-metadata --excel-file FILE.xlsx
```

**How it works:**
- Reads Excel file with ID, Age, Sex, Class columns
- Matches .wav files to Excel by ID in filename
- Uses Age and Sex as additional features
- Predicts Class values from Excel (not folder names)

**Example:**
```bash
python train.py \
    --data-dir data/task1/training \
    --use-metadata \
    --excel-file data/task1/sand_task_1.xlsx \
    --model-type logistic_regression
```

**You can still select specific folders:**
```bash
python train.py \
    --data-dir data/task1/training \
    --subdirs phonationA rhythmKA \
    --use-metadata \
    --excel-file data/task1/sand_task_1.xlsx
```

**Result:** Model learns to predict Class 1-5 from audio features + Age + Sex

---

### Mode 3: Temporal Feature Extraction

**Goal:** Capture how audio changes over time

**Command structure:**
```bash
python train.py --data-dir DIRECTORY --use-temporal [options]
```

**How it works:**
- Splits audio into short frames (10-25ms)
- Extracts features from each frame
- Includes velocity (delta) and acceleration (delta-delta)
- Aggregates frames into fixed-size vector

**Example:**
```bash
python train.py \
    --data-dir data/task1/training \
    --subdirs phonationA phonationE \
    --use-temporal \
    --frame-length 20.0 \
    --hop-length 10.0 \
    --context-frames 1
```

**Can combine with metadata:**
```bash
python train.py \
    --data-dir data/task1/training \
    --use-temporal \
    --frame-length 20.0 \
    --use-metadata \
    --excel-file data/task1/sand_task_1.xlsx
```

---

## Understanding Features

### Standard Features (Default)

When you run without `--use-temporal`:

**Extracted features (66 total):**
1. **13 MFCCs** - Mean and Std (26 features)
2. **Spectral Centroid** - Mean and Std (2 features)
3. **Spectral Rolloff** - Mean and Std (2 features)
4. **Spectral Contrast (7 bands)** - Mean and Std (14 features)
5. **Zero Crossing Rate** - Mean and Std (2 features)
6. **Chroma (12 pitch classes)** - Mean and Std (24 features)

**Total:** 66 features per audio file

**With metadata:** 66 audio + 2 metadata (Age, Sex) = **68 features**

---

### Temporal Features (with `--use-temporal`)

When you run with `--use-temporal`:

**Frame-based extraction:**
1. Audio split into frames (e.g., 25ms windows every 10ms)
2. MFCCs extracted per frame
3. Delta features (velocity) computed
4. Delta-delta features (acceleration) computed
5. Optional: Context frames stacked
6. Aggregated with mean + std

**Feature dimensions:**

| Configuration | Per Frame | Aggregated |
|---------------|-----------|------------|
| MFCCs only | 13 | 26 |
| + Deltas (default) | 26 | 52 |
| + Delta-deltas (default) | 39 | 78 |
| + Context=1 | 117 | 234 |
| + Context=2 | 195 | 390 |

**With metadata:** Temporal features + 2 metadata features

---

## Troubleshooting

### Installation Issues

**Problem:** `pip install -r requirements.txt` fails

**Solutions:**
```bash
# Upgrade pip first
pip install --upgrade pip

# Try installing packages one by one
pip install librosa
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install openpyxl
```

---

**Problem:** Virtual environment won't activate

**Solution (macOS/Linux):**
```bash
# Make sure you're in project directory
cd /path/to/setup

# Create fresh venv
python3 -m venv venv

# Activate
source venv/bin/activate
```

**Solution (Windows):**
```bash
# Create fresh venv
python -m venv venv

# Activate
venv\Scripts\activate
```

---

### Data Issues

**Problem:** "No .wav files found"

**Solutions:**
1. Check your path is correct:
   ```bash
   ls data/task1/training/phonationA  # Should show .wav files
   ```

2. Make sure you're using correct directory:
   ```bash
   # Wrong - pointing to a file
   --data-dir data/task1/training/phonationA/file.wav
   
   # Correct - pointing to directory containing folders
   --data-dir data/task1/training
   ```

3. Check subdirectory names are spelled correctly:
   ```bash
   # Wrong
   --subdirs phonationA PhonationE
   
   # Correct (case-sensitive!)
   --subdirs phonationA phonationE
   ```

---

**Problem:** "Only one class found"

**Cause:** All your audio files have the same label

**Solutions:**

**For standard mode:** Use at least 2 subdirectories
```bash
# Wrong - only one folder
--subdirs phonationA

# Correct - multiple folders
--subdirs phonationA phonationE
```

**For metadata mode:** Check your Excel file
```bash
# Open Excel file and check Class column
# Make sure it has at least 2 different values (e.g., 1, 2, 3, 4, 5)
# If all rows have the same Class value, model can't learn
```

---

**Problem:** "Files without metadata" warning

**Cause:** .wav files don't match IDs in Excel file

**Solutions:**

1. Check filename format:
   ```
   Filenames should contain the ID from Excel
   Example Excel: ID = 123
   Example filename: 123_phonationA.wav or phonationA_123.wav
   ```

2. Verify Excel file:
   ```bash
   # Open sand_task_1.xlsx
   # Check that ID column values appear in .wav filenames
   ```

3. This warning is OK if most files match (some missing is acceptable)

---

### Performance Issues

**Problem:** Low accuracy (< 50%)

**Causes and solutions:**

1. **Imbalanced classes**
   ```
   Class 1:   6 samples
   Class 5: 107 samples
   
   Solution: Use more data or try Random Forest (handles imbalance better)
   ```

2. **Not enough data**
   ```
   Solution: Include more subdirectories or use all available data
   --data-dir data/task1/training  # Use all folders
   ```

3. **Wrong model type**
   ```
   Try different models:
   --model-type random_forest
   --model-type logistic_regression
   --model-type svm
   ```

4. **Overfitting** (Train accuracy >> Test accuracy)
   ```
   Training accuracy: 90%
   Test accuracy: 30%
   
   Solutions:
   - Use more data
   - Try simpler model (logistic_regression)
   - Increase test-size to 0.3
   ```

---

**Problem:** Training is very slow

**Solutions:**

1. Reduce audio processing:
   ```bash
   --sample-rate 16000      # Lower sample rate
   --n-mfcc 13              # Use default (not 40)
   ```

2. Use fewer subdirectories for testing:
   ```bash
   --subdirs phonationA phonationE  # Just 2 instead of all 8
   ```

3. Disable temporal features for faster training:
   ```bash
   # Remove --use-temporal flag
   ```

4. Use faster model:
   ```bash
   --model-type logistic_regression  # Faster than SVM or Random Forest
   ```

---

### Error Messages

**Error:** `ModuleNotFoundError: No module named 'librosa'`

**Solution:** Install requirements
```bash
pip install -r requirements.txt
```

---

**Error:** `Excel file not found`

**Solution:** Check path
```bash
# Make sure file exists
ls data/task1/sand_task_1.xlsx

# Use correct path in command
--excel-file data/task1/sand_task_1.xlsx
```

---

**Error:** `--excel-file required when --use-metadata is set`

**Solution:** Add excel file path
```bash
python train.py \
    --data-dir data/task1/training \
    --use-metadata \
    --excel-file data/task1/sand_task_1.xlsx  # Add this!
```

---

## Advanced Usage

### Comparing Different Configurations

Train multiple models and compare results:

```bash
# Model 1: Standard features
python train.py \
    --data-dir data/task1/training \
    --subdirs phonationA phonationE phonationI \
    --model-type random_forest \
    --model-name model1_standard \
    --save-plots

# Model 2: Temporal features
python train.py \
    --data-dir data/task1/training \
    --subdirs phonationA phonationE phonationI \
    --use-temporal \
    --model-type random_forest \
    --model-name model2_temporal \
    --save-plots

# Model 3: With metadata
python train.py \
    --data-dir data/task1/training \
    --use-metadata \
    --excel-file data/task1/sand_task_1.xlsx \
    --model-type logistic_regression \
    --model-name model3_metadata \
    --save-plots
```

Then compare test accuracy to see which performs best!

---

### Testing Different Test Types

Compare which audio test is best for predicting Class:

```bash
# Phonation tests only
python train.py \
    --data-dir data/task1/training \
    --subdirs phonationA phonationE phonationI phonationO phonationU \
    --use-metadata \
    --excel-file data/task1/sand_task_1.xlsx \
    --model-name phonation_tests

# Rhythm tests only  
python train.py \
    --data-dir data/task1/training \
    --subdirs rhythmKA rhythmPA rhythmTA \
    --use-metadata \
    --excel-file data/task1/sand_task_1.xlsx \
    --model-name rhythm_tests

# All tests combined
python train.py \
    --data-dir data/task1/training \
    --use-metadata \
    --excel-file data/task1/sand_task_1.xlsx \
    --model-name all_tests
```

---

### Hyperparameter Tuning

Try different settings to find the best configuration:

**Frame length variations:**
```bash
python train.py --data-dir DATA --use-temporal --frame-length 15.0 --model-name frame_15
python train.py --data-dir DATA --use-temporal --frame-length 20.0 --model-name frame_20
python train.py --data-dir DATA --use-temporal --frame-length 25.0 --model-name frame_25
```

**Context variations:**
```bash
python train.py --data-dir DATA --use-temporal --context-frames 0 --model-name context_0
python train.py --data-dir DATA --use-temporal --context-frames 1 --model-name context_1
python train.py --data-dir DATA --use-temporal --context-frames 2 --model-name context_2
```

**Model comparisons:**
```bash
python train.py --data-dir DATA --model-type random_forest --model-name rf
python train.py --data-dir DATA --model-type svm --model-name svm
python train.py --data-dir DATA --model-type logistic_regression --model-name lr
```

---

### Understanding Output

When training completes, you'll see:

```
================================================================================
TRAINING SUMMARY
================================================================================
Total samples: 272
Number of features: 72
Number of classes: 5
Classes: 1, 2, 3, 4, 5

Test Accuracy: 0.4545
Cross-validation Accuracy: 0.4520 (+/- 0.1357)

Model saved to: models/logistic_regression_20251009_143022.pkl
================================================================================
```

**What these mean:**

- **Total samples:** How many audio files were used
- **Number of features:** Size of feature vector
- **Number of classes:** How many categories to predict
- **Test Accuracy:** Performance on unseen data (0-1 scale, higher is better)
- **Cross-validation Accuracy:** Average across 5 different splits (more reliable)
- **+/- value:** Uncertainty in CV score (smaller is more stable)

**Good results:**
- Test accuracy > 0.70 (70%)
- CV accuracy close to test accuracy (within 0.05)
- Small +/- value (< 0.10)

**Poor results:**
- Test accuracy < 0.50 (50%)
- Large gap between train and test accuracy (overfitting)
- Large +/- value (> 0.20) (unstable)

---

### Using Trained Models

After training, use your model for predictions:

```bash
# Predict on a single file
python predict.py \
    --model models/your_model.pkl \
    --audio path/to/audio.wav

# Predict on a whole directory
python predict.py \
    --model models/your_model.pkl \
    --audio-dir path/to/folder \
    --output predictions.csv
```

---

## Getting Help

**See all available options:**
```bash
python train.py --help
```

**Check Python version:**
```bash
python --version
```

**Check if packages are installed:**
```bash
pip list | grep librosa
pip list | grep scikit-learn
```

**Test if everything works:**
```bash
# Quick test with minimal data
python train.py \
    --data-dir data/task1/training \
    --subdirs phonationA phonationE \
    --test-size 0.5
```

---

## Summary of Common Commands

**Basic training (folder-based):**
```bash
python train.py --data-dir data/task1/training --subdirs phonationA phonationE
```

**Metadata mode (predict Class 1-5):**
```bash
python train.py --data-dir data/task1/training --use-metadata --excel-file data/task1/sand_task_1.xlsx
```

**Temporal features:**
```bash
python train.py --data-dir data/task1/training --subdirs phonationA phonationE --use-temporal
```

**Everything combined:**
```bash
python train.py --data-dir data/task1/training --use-temporal --use-metadata --excel-file data/task1/sand_task_1.xlsx --save-plots
```

**Try different model:**
```bash
python train.py --data-dir data/task1/training --subdirs phonationA phonationE --model-type logistic_regression
```

---

## Tips for Success

1. **Start simple** - Use default settings first
2. **Use 2-3 classes** initially to understand the system
3. **Always use --save-plots** to visualize results
4. **Compare multiple models** to find the best
5. **Check class balance** - roughly equal samples per class is ideal
6. **More data = better results** - use all available subdirectories when possible
7. **Cross-validation score is more reliable** than single test accuracy
8. **Temporal features** usually improve results for speech/voice data
9. **Logistic regression** works well for Class (1-5) prediction
10. **Give models meaningful names** to track experiments

---

**Ready to start? Run this test command:**

```bash
python train.py --data-dir data/task1/training --subdirs phonationA phonationE --save-plots
```

Good luck! 🎉

