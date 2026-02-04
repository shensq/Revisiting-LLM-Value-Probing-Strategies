# Revisiting LLM Value Probing Strategies: Are They Robust and Expressive?

This repository contains the code and data for the EMNLP 2025 paper:

**"Revisiting LLM Value Probing Strategies: Are They Robust and Expressive?"**  
*Siqi Shen, Mehar Singh, Lajanugen Logeswaran, Moontae Lee, Honglak Lee, Rada Mihalcea*

## Overview

This project systematically evaluates three widely-used methods for probing value orientations in Large Language Models (LLMs):

1. **Token Logit Method**: Uses next-token probabilities for answer choices
2. **Sequence Perplexity Method**: Uses perplexity of complete answer sequences
3. **Text Generation Method**: Uses sampled free-text outputs

We assess these methods across two key dimensions:
- **Robustness**: Stability under input perturbations (prompt variations, selection bias)
- **Expressiveness**: Responsiveness to demographic context and correlation with value-based actions

## Key Findings

- All probing methods show significant sensitivity to input formatting and selection bias
- Sequence perplexity method tends to be most robust to perturbations
- Value representations can be steered by demographic prompting, but text generation captures this poorly
- Weak correlation (0.1-0.3) between probed values and action preferences in value-related scenarios
- Larger models demonstrate greater stability across all methods

## Repository Structure

```
.
├── data/                           # Data files and inputs
│   ├── question_options.csv        # 206 WVS questions used in experiments
│   ├── wvs_survey_by_country.json  # Human survey results by country
│   ├── input_dict.json             # Processed input dictionary
│   ├── base_variations/            # Inputs for Experiment 1 (robustness)
│   ├── demographic_prompting/      # Inputs for Experiment 2 (expressiveness)
│   ├── templates/                  # Prompt templates (3 styles)
│   └── action_agreement/           # Synthesized scenarios for Experiment 3
├── src/                            # Source code
│   ├── construct_inputs.py         # Generate input files for all experiments
│   ├── run_inference.py            # Run model inference
│   ├── inference_utils.py          # Inference helper functions
│   ├── parse_generation.py         # Parse generated text outputs
│   ├── compute_metrics.py          # Calculate metrics and generate plots
│   ├── eval_utils.py               # Evaluation utilities
│   └── run_action_scoring.py       # Score actions for Experiment 3
├── scripts/                        # Execution scripts
│   ├── run_exp1.sh                 # Run Experiment 1 (robustness)
│   ├── run_exp2.sh                 # Run Experiment 2 (demographic prompting)
│   ├── run_exp3.sh                 # Run Experiment 3 (action agreement)
│   └── sbatch.sh                   # SLURM batch script
├── results/                        # Output directory for inference results
└── requirements.txt                # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for running models)
- Hugging Face account with access to gated models (e.g., Llama)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/shensq/Revisiting-LLM-Value-Probing-Strategies.git
cd value_alignment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add Hugging Face access token to `credentials.py` in the `src/` directory:
```python
# src/credentials.py
access_token = "your_huggingface_token_here"
```


## Dataset

This project uses the **World Values Survey (WVS) Wave 7**, which contains responses from 129K+ participants across multiple countries on 13 subjective topic areas including:
- Social and religious values
- Happiness and well-being
- Trust and organizational membership
- Economic values
- Political interest and participation
- Ethical values and norms

We selected 206 questions from the survey, filtering out:
- Questions dependent on other questions
- Questions with respondent-specific demographic information
- Non-ordinal questions (for certain experiments)

### Countries Studied
- United States
- Germany
- Czech Republic
- China
- Mexico
- Egypt

## Usage

### Step 1: Construct Input Files

Generate all input variations for the experiments:

```bash
python src/construct_inputs.py
```

This creates:
- **Base variations**: baseline, num (numeric labels), reverse (reversed order), lower (lowercase), shuffle (shuffled)
- **Demographic prompting**: Country-specific versions of each variation
- **Action agreement**: Scenario-action pairs for value correlation

### Step 2: Run Experiments

#### Experiment 1: Robustness to Prompt Variations and Selection Bias

Tests how stable value representations are across different prompt styles and option label variations.

```bash
# For single model
python src/run_inference.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --input_path data/base_variations/baseline.json \
    --num_return_sequences 10 \
    --save_dir results/

# For all models (using SLURM)
bash scripts/run_exp1.sh
```

**Models tested**: Llama-3.1/3.2 (3B, 8B, 70B), Qwen2.5 (3B, 7B, 14B, 72B), Mistral-7B, Falcon-7B, Bloomz-7B

#### Experiment 2: Demographic Prompting

Tests whether value representations align better with specific countries when given demographic context.

```bash
# For single model and country
python src/run_inference.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --input_path data/demographic_prompting/China_baseline.json \
    --num_return_sequences 10 \
    --save_dir results/

# For all models and countries
bash scripts/run_exp2.sh
```

#### Experiment 3: Value-Action Agreement

Tests correlation between probed values and action ratings in value-related scenarios.

```bash
# Score actions for each model
python src/run_action_scoring.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --input_path data/action_agreement/scoring_samples.jsonl \
    --save_dir results/

# For all models
bash scripts/run_exp3.sh
```

### Step 3: Parse Generated Text

Extract answer labels from free-text generations:

```bash
python src/parse_generation.py
```

This processes all result files and adds a `parsed_generation` field.

### Step 4: Compute Metrics and Generate Plots

Calculate metrics and create visualizations:

```bash
python src/compute_metrics.py
```

This generates PDF plots in `writeup/` directory.


## Probing Methods Explained

### 1. Token Logit Method (`option_probs`)

Extracts the logits of valid answer tokens (e.g., "A", "B", "C") from the first generated token:

```python
p_token = softmax(logits)
```


### 2. Sequence Perplexity Method (`seq_probs`)

Computes perplexity of the complete answer sequence (e.g., "A. Strongly agree"):

```python
p_seq = perplexity^(-1) / Σ(perplexity_i^(-1))
```


### 3. Text Generation Method (`text_probs`)

Samples multiple outputs and counts answer frequencies:

```python
p_text = count(option) / num_samples
```


## Prompt Templates

Three prompt styles are used to test robustness:

1. **Default**: Standard instruction with question and options
2. **Prefixed**: Adds affirmative starter ("Certainly! I would select option")
3. **One-shot**: Includes example question-answer pair

Templates are defined in `data/templates/template{0,1,2}.json`

## Citation

```bibtex
@inproceedings{shen-etal-2025-revisiting,
    title = "Revisiting LLM Value Probing Strategies: Are They Robust and Expressive?",
    author = "Shen, Siqi  and
      Singh, Mehar  and
      Logeswaran, Lajanugen  and
      Lee, Moontae  and
      Lee, Honglak  and
      Mihalcea, Rada",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.7/",
    doi = "10.18653/v1/2025.emnlp-main.7",
    pages = "131--145",
    ISBN = "979-8-89176-332-6",
}
```
