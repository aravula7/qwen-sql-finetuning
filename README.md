# Qwen2.5-3B Text-to-SQL Fine-Tuning

Fine-tuned Qwen2.5-3B-Instruct for PostgreSQL text-to-SQL generation over an e-commerce + subscriptions + ML predictions schema. Trained using QLoRA with comprehensive evaluation against baseline and commercial API models.

## Results Summary

Evaluated on 100 test examples with parseable SQL (validated via sqlglot) as the primary metric and exact match as secondary.

| Model | Parseable SQL | Exact Match | Mean Latency (s) | P50 (s) | P95 (s) |
|-------|---------------|-------------|------------------|---------|---------|
| **qwen_finetuned_fp16_strict** | **1.00** | **0.15** | **0.433** | 0.427 | 0.736 |
| qwen_finetuned_int8_strict | 0.99 | 0.20 | 2.152 | 2.541 | 3.610 |
| qwen_baseline_fp16 | 1.00 | 0.09 | 0.405 | 0.422 | 0.624 |
| qwen_finetuned_fp16 | 0.93 | 0.13 | 0.527 | 0.711 | 0.739 |
| qwen_finetuned_int8 | 0.93 | 0.13 | 2.672 | 3.454 | 3.623 |
| gpt-4o-mini | 1.00 | 0.04 | 1.616 | 1.551 | 2.820 |
| claude-3.5-haiku | 0.99 | 0.07 | 1.735 | 1.541 | 2.697 |

**Key Findings:**
- Strict SQL-only prompting (no prose, no markdown) dramatically improved reliability from 93% to 100% parseable rate
- Fine-tuned FP16 model matches baseline reliability while improving exact match accuracy by 67% (0.09 → 0.15)
- INT8 quantization maintains accuracy with 50% memory reduction but shows 5x latency increase in this evaluation setup
- Fine-tuned model outperforms commercial APIs on exact match while maintaining comparable speed

## Architecture

![Workflow Architecture](images/workflow_architecture_diagram.svg)

**Note:** If the diagram doesn't display properly (transparent background in some viewers), view the [PNG version](images/workflow_architecture_diagram.png) instead.

**System Overview:**

- **Data Pipeline**: 500 SQL examples split into training (350), validation (50), and test (100) sets with schema context included in each prompt
- **Training**: QLoRA 4-bit training with paged_adamw_8bit optimizer for memory-efficient fine-tuning (4 epochs, ~4 minutes on A100)
- **Model Outputs**: Three variants produced - FP16 full precision (6GB), INT8 quantized (3GB), and LoRA adapter only (50MB)
- **Evaluation**: 7 models tested on 100 examples using greedy decoding with sqlglot validation for parseability
- **Tracking**: Complete experiment lifecycle logged to MLflow (DagsHub) including training metrics, hyperparameters, and evaluation results
- **Deployment**: All model variants hosted on HuggingFace Hub for public access and reproduction

## Visualizations

### Model Performance Comparison

<table>
<tr>
<td width="50%">

**2D Comparison**

![Results Comparison](images/results_comparison.png)

Side-by-side comparison of parseable SQL rate (primary metric) and exact match accuracy (secondary metric) across all 7 evaluated models.

</td>
<td width="50%">

**3D Multi-Metric View**

![3D Model Comparison](images/model_comparison_3d.png)

Three-dimensional visualization showing parseable rate, exact match, and speed (inverse latency) for comprehensive model comparison.

</td>
</tr>
<tr>
<td width="50%">

**2D Trade-off Analysis**

![Accuracy vs Latency](images/accuracy_vs_latency.png)

Traditional scatter plot showing the accuracy-latency trade-off. Lower-right quadrant is optimal (high accuracy, low latency).

</td>
<td width="50%">

**3D Trade-off Space**

![3D Accuracy-Latency](images/accuracy_latency_3d.png)

Advanced 3D scatter plot incorporating model size (bubble size) alongside accuracy and latency dimensions for complete trade-off analysis.

</td>
</tr>
</table>

## Training Details

### Dataset

Custom SQL generation dataset covering e-commerce + subscriptions + ML predictions schema:

**Schema Tables:**
- `customers`: Customer dimension (customer_id, email, region, state)
- `products`: Product catalog (product_id, sku, category, base_price_usd)
- `orders`: Order headers (order_id, customer_id, order_timestamp)
- `order_items`: Line items (order_item_id, order_id, product_id, quantity, unit_price_usd)
- `subscriptions`: Subscription events (subscription_id, customer_id, plan_name, status)
- `churn_predictions`: ML churn predictions (customer_id, churn_probability, churn_flag)
- `forecast_predictions`: ML demand forecasts (product_id, region, predicted_units)

**Dataset Distribution:**

| Category | Train | Val | Test | Total |
|----------|-------|-----|------|-------|
| Single-table queries | 70 | 10 | 20 | 100 |
| Two-table joins | 53 | 8 | 14 | 75 |
| Revenue analytics | 53 | 8 | 14 | 75 |
| Time-series queries | 35 | 5 | 10 | 50 |
| Churn analytics | 35 | 5 | 10 | 50 |
| Forecast analytics | 35 | 5 | 10 | 50 |
| Subscription analytics | 35 | 5 | 10 | 50 |
| Window functions | 18 | 2 | 5 | 25 |
| Complex multi-table | 11 | 1 | 3 | 15 |
| Edge cases | 5 | 1 | 4 | 10 |
| **Total** | **350** | **50** | **100** | **500** |

**Dataset Strategy:**

Each training example includes schema context (Option A approach), teaching the model to generate SQL given schema + natural language query. This matches production usage where relevant table schemas are provided via RAG or documentation.

The 10 categories ensure comprehensive coverage of query complexity from simple single-table operations to advanced window functions and multi-table joins with CTEs. Realistic values (dates, states, prices) and proper PostgreSQL syntax (DATE_TRUNC, EXTRACT, etc.) are used throughout.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen2.5-3B-Instruct |
| Method | QLoRA (4-bit quantization) |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Optimizer | paged_adamw_8bit |
| Learning Rate | 2e-4 |
| Weight Decay | 0.01 |
| Epochs | 4 |
| Batch Size | 16 (4 per device × 4 gradient accumulation) |
| Max Sequence Length | 2048 |
| Warmup Steps | 100 |
| Training Time | ~4 minutes on A100 |
| Trainable Parameters | 29.9M (1.73% of total) |

**Optimizer Choice:**

`paged_adamw_8bit` reduces optimizer memory footprint by 4x (36GB → 9GB) through 8-bit quantization and CPU paging, enabling efficient fine-tuning on consumer GPUs while maintaining training quality.

## Evaluation Methodology

### Metrics

**Primary Metric: Parseable SQL**
- SQL query successfully parsed by `sqlglot` library
- Validates syntactic correctness and PostgreSQL compatibility
- Most important for production deployment

**Secondary Metric: Exact Match**
- Character-level exact match after normalization
- Strict but limited (multiple valid SQL queries can answer same question)
- Useful for measuring training convergence

**Performance Metrics:**
- Mean Latency: Average inference time per query
- P50 Latency: Median inference time (typical performance)
- P95 Latency: 95th percentile (tail latency)

### Evaluation Setup

- Test set: 100 examples covering all 10 query categories
- Greedy decoding (no sampling) for reproducibility
- Hardware: NVIDIA A100 GPU
- Validation: sqlglot parser with PostgreSQL dialect

### Model Variants

**Baseline:**
- `qwen_baseline_fp16`: Unmodified Qwen2.5-3B-Instruct (0-shot)

**Fine-tuned (standard prompting):**
- `qwen_finetuned_fp16`: Merged LoRA adapters, full precision
- `qwen_finetuned_int8`: Quantized to INT8

**Fine-tuned (strict SQL-only prompting):**
- `qwen_finetuned_fp16_strict`: SQL-only system instruction
- `qwen_finetuned_int8_strict`: SQL-only + INT8 quantization

**Commercial APIs:**
- `gpt-4o-mini`: OpenAI's efficient model
- `claude-3.5-haiku`: Anthropic's fast model

### Key Insights

1. **Strict prompting is critical**: Adding "Return ONLY the PostgreSQL query. Do NOT include explanations, markdown, or commentary" improved parseable rate from 93% to 100%

2. **Fine-tuning improves accuracy**: Exact match increased from 9% (baseline) to 15% (fine-tuned), a 67% improvement

3. **Quantization trade-offs**:
   - INT8 maintains accuracy (20% exact match, best across all models)
   - But shows 5x latency increase (0.43s → 2.15s) in this setup
   - Memory footprint reduced by 50% (6GB → 3GB)

4. **Competitive with APIs**: Fine-tuned model achieves 4x better exact match than GPT-4o-mini while maintaining similar latency

## Installation

### Requirements

```bash
# Clone repository
git clone https://github.com/aravula7/qwen-sql-finetuning.git
cd qwen-sql-finetuning

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=sk-proj-your_openai_api_key_here
ANTHROPIC_API_KEY=sk-ant-your_anthropic_api_key_here
DAGSHUB_TOKEN=your_dagshub_token_here
HF_TOKEN=your_huggingface_token_here
```

## Usage

### Basic Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model from HuggingFace
repo_id = "aravula7/qwen-sql-finetuning"
tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder="fp16")
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    subfolder="fp16",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Prepare prompt
system = "Return ONLY the PostgreSQL query. Do NOT include explanations, markdown, code fences, or commentary."
schema = """Table: customers (customer_id, email, region, state)
Table: orders (order_id, customer_id, order_timestamp)
Table: order_items (order_item_id, order_id, product_id, quantity, unit_price_usd)"""
request = "Show total revenue by region for orders in 2024"

prompt = f"""{system}

Schema:
{schema}

Request:
{request}
"""

# Generate SQL
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
# Extract SQL after prompt
sql = sql.split("Request:")[-1].strip()
print(sql)
```

### Production-Ready Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sqlglot
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLGenerator:
    def __init__(self, model_path: str = "aravula7/qwen-sql-finetuning", variant: str = "fp16"):
        """
        Initialize SQL generator.
        
        Args:
            model_path: HuggingFace repo ID
            variant: Model variant ("fp16", "int8", or "lora_adapter")
        """
        self.repo_id = model_path
        self.variant = variant
        
        logger.info(f"Loading {variant} model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder=variant)
        
        load_kwargs = {
            "subfolder": variant,
            "device_map": "auto"
        }
        
        if variant == "fp16":
            load_kwargs["torch_dtype"] = torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        logger.info("Model loaded successfully")
    
    def generate_sql(
        self,
        request: str,
        schema: str,
        max_tokens: int = 256,
        validate: bool = True
    ) -> Dict[str, any]:
        """
        Generate SQL query from natural language request.
        
        Args:
            request: Natural language query
            schema: Database schema context
            max_tokens: Maximum tokens to generate
            validate: Whether to validate SQL with sqlglot
        
        Returns:
            Dictionary with 'sql', 'valid', and optional 'error' keys
        """
        system = "Return ONLY the PostgreSQL query. Do NOT include explanations, markdown, code fences, or commentary."
        
        prompt = f"""{system}

Schema:
{schema}

Request:
{request}
"""
        
        try:
            # Generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            sql = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            sql = sql.split("Request:")[-1].strip()
            
            # Clean SQL
            sql = sql.replace("```sql", "").replace("```", "").strip()
            
            result = {
                "sql": sql,
                "valid": True,
                "error": None
            }
            
            # Validate if requested
            if validate:
                try:
                    sqlglot.parse_one(sql, dialect="postgres")
                    logger.info("SQL validation passed")
                except Exception as e:
                    result["valid"] = False
                    result["error"] = f"SQL validation failed: {str(e)}"
                    logger.warning(result["error"])
            
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return {
                "sql": None,
                "valid": False,
                "error": f"Generation failed: {str(e)}"
            }

# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = SQLGenerator(variant="fp16")
    
    # Define schema
    schema = """Table: customers (customer_id, email, region, state)
Table: orders (order_id, customer_id, order_timestamp)
Table: order_items (order_item_id, order_id, product_id, quantity, unit_price_usd)"""
    
    # Generate SQL
    request = "Show total revenue by region for orders in 2024"
    result = generator.generate_sql(request, schema)
    
    if result["valid"]:
        print(f"Generated SQL:\n{result['sql']}")
    else:
        print(f"Error: {result['error']}")
```

### Load INT8 Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "aravula7/qwen-sql-finetuning"
tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder="int8")
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    subfolder="int8",
    device_map="auto"
)
```

### Load LoRA Adapter Only

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
repo_id = "aravula7/qwen-sql-finetuning"
model = PeftModel.from_pretrained(base_model, repo_id, subfolder="lora_adapter")
```

## Repository Structure

```
qwen-sql-finetuning/
├── Qwen_SQL_FineTuning.ipynb    # Training & Evaluation Notebook
├── data/
│   ├── train.jsonl              # Training set (350 examples)
│   ├── validation.jsonl         # Validation set (50 examples)
│   └── test.jsonl               # Test set (100 examples)
├── images/
│   ├── workflow_architecture_diagram.svg  # System architecture
│   ├── results_comparison.png             # 2D performance comparison
│   ├── model_comparison_3d.png           # 3D multi-metric view
│   ├── accuracy_latency_3d.png           # 3D trade-off analysis
│   └── accuracy_vs_latency.png           # 2D trade-off scatter
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

**Models hosted separately on HuggingFace:**
- `fp16/` - Merged FP16 model (6GB)
- `int8/` - Quantized INT8 model (3GB)
- `lora_adapter/` - LoRA adapter only (50MB)

## MLflow Tracking

All training runs and evaluation metrics are tracked with MLflow on DagsHub:

**Logged Metrics:**
- Training: loss, perplexity (per epoch)
- Evaluation: parseable_rate, exact_match_rate, latency (P50, P95, mean)
- Model metadata: hyperparameters, dataset splits, model sizes

**Access Experiments:**

Visit the DagsHub repository to view interactive dashboards, metric comparisons, and run artifacts.

## Citation

If you use this model or methodology in your research, please cite:

```bibtex
@misc{qwen-sql-finetuning-2026,
  author = {Anirudh Reddy Ravula},
  title = {Qwen2.5-3B Text-to-SQL Fine-Tuning for PostgreSQL},
  year = {2026},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/aravula7/qwen-sql-finetuning}},
  note = {Fine-tuned with QLoRA for e-commerce SQL generation}
}
```

## License

This project is licensed under the MIT License. The fine-tuned model is a derivative of Qwen2.5-3B-Instruct and inherits its license terms.

## Links

- **HuggingFace Model**: https://huggingface.co/aravula7/qwen-sql-finetuning
- **GitHub Repository**: https://github.com/aravula7/qwen-sql-finetuning
- **DagsHub Experiments**: https://dagshub.com/aravula7/qwen-sql-finetuning
- **Base Model**: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct

## Acknowledgments

- Base model: Qwen2.5 3B Instruct (Alibaba Cloud)
- Training framework: HuggingFace Transformers, PEFT, TRL
- Tracking: MLflow/DagsHub
- Evaluation: sqlglot for SQL parsing validation