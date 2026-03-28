# SageMaker Python SDK v3.4.0 Reference

## Overview
The SageMaker Python SDK v3.4.0 is a modular metapackage with four sub-packages: `sagemaker-core`, `sagemaker-train`, `sagemaker-serve`, and `sagemaker-mlops`. Supports Python 3.9-3.12.

---

## 1. Core Infrastructure (`sagemaker-core`)
- **Resource Management**: Object-oriented, class-based interface for all SageMaker resources
- **Resource Chaining**: Pass outputs of one resource as inputs to another
- **Automatic State Management**: Handles polling and state transitions (e.g., waiting for jobs)
- **Intelligent Defaults**: Auto-fills IAM roles, VPC configs, etc.
- **Full Type Hints**: IDE auto-completion and inline docs

---

## 2. Model Training (`sagemaker-train`)

### Unified `ModelTrainer` Class
Replaces all V2 framework-specific estimators with a single class.

### Supported Frameworks
PyTorch, TensorFlow, MXNet, Scikit-learn, XGBoost, HuggingFace, custom containers

### Key Features
- **Custom Training Containers**: Bring-your-own Docker images
- **Distributed Training**: Multi-instance training support
- **Local Training**: Train locally before scaling to cloud
- **Hyperparameter Tuning**: Automatic model tuning / HPO
- **InputData Class**: Structured S3 data source configuration
- **JumpStart Training**: Pre-built recipes for foundation models
- **Built-in Algorithms**: Linear Learner, XGBoost, K-Means, PCA, etc.

### Fine-Tuning (New in V3)
- **SFTTrainer** - Supervised fine-tuning
- **DPOTrainer** - Direct Preference Optimization
- **RLAIFTrainer** - Reinforcement Learning from AI Feedback
- **RLVRTrainer** - Reinforcement Learning from Verifiable Rewards
- LoRA and full fine-tuning support
- MLflow integration with real-time metrics
- Built-in evaluation across 11 benchmarks

### Example: Training
```python
from sagemaker.train import ModelTrainer, InputData

trainer = ModelTrainer(
    training_image="pytorch-training:2.0-gpu-py310",
    source_code="train.py",
    instance_type="ml.m5.large",
    instance_count=1,
    hyperparameters={"epochs": 10, "lr": 0.001},
)

trainer.train(
    input_data=InputData(s3_uri="s3://bucket/data")
)
```

---

## 3. Model Serving (`sagemaker-serve`)

### Unified `ModelBuilder` Class
Replaces V2 Model/Predictor and framework-specific model classes.

### Deployment Options
- **Real-Time Endpoints**: Persistent endpoints for synchronous predictions
- **Batch Transform**: Inference on entire datasets without persistent endpoints
- **JumpStart Deployment**: One-click deploy of pre-trained foundation models
- **HuggingFace Models**: Direct deployment support
- **Local Mode**: Test inference locally first
- **In-Process Mode**: Run inference in same Python process for debugging
- **Deploy to SageMaker or Bedrock**: Fine-tuned models go to either

### Additional Features
- **Inference Spec**: Custom pre/post-processing logic
- **Model Optimization**: Compilation, quantization
- **Inference Pipelines**: Chain multiple models in one endpoint
- **SparkML Serving**: Host SparkML models with MLeap
- **BYO Model**: Deploy pre-trained models from any source

### Example: Deployment
```python
from sagemaker.serve import ModelBuilder

builder = ModelBuilder(
    model="my-model",
    schema_builder=SchemaBuilder(sample_input, sample_output),
)

model = builder.build()
predictor = model.deploy(instance_type="ml.m5.large", initial_instance_count=1)
result = predictor.predict(data)
```

---

## 4. MLOps & Pipelines (`sagemaker-mlops`)

### SageMaker Pipelines
Define, execute, and manage ML workflows as DAGs.

### 13+ Step Types
- Training / AutoML training steps
- Model creation and registration steps
- Processing jobs (Scikit-learn, PyTorch)
- Batch transform steps
- AWS Lambda invocation steps
- Amazon EMR cluster operations
- Notebook job execution steps
- Quality check steps (data quality, model quality)
- Bias Check steps (SageMaker Clarify)
- Custom callback steps
- Failure steps (conditional error handling)

### Pipeline Features
- **Conditional Execution**: Branch logic based on conditions
- **Selective Execution**: Run specific parts only
- **Retry Policies**: Automatic retries with error handling
- **Parallelism**: Run steps concurrently
- **Experiment Tracking**: Integrate with SageMaker Experiments
- **Model Registry**: Version, manage, and approve models
- **EventBridge Integration**: Trigger from external events
- **Model Monitoring**: Data quality, model quality, bias/drift detection

---

## 5. Integrations
| Service | Use Case |
|---------|----------|
| MLflow | Real-time metrics tracking during fine-tuning |
| Amazon Bedrock | Deploy fine-tuned models |
| Amazon EventBridge | Event-driven pipeline triggers |
| Amazon EMR | Spark-based processing in pipelines |
| AWS Lambda | Serverless functions in pipelines |
| Apache Airflow | Workflow orchestration |

---

## 6. Modular Installation
```bash
pip install sagemaker           # Full package
pip install sagemaker[train]    # Training only
pip install sagemaker[serve]    # Serving only
pip install sagemaker[mlops]    # MLOps only
pip install sagemaker[all]      # Everything
```

---

## Quick Reference: V2 vs V3

| V2 (Legacy) | V3 (Current) |
|---|---|
| `Estimator` + framework subclasses | Unified `ModelTrainer` |
| `Model` + framework subclasses | Unified `ModelBuilder` |
| Monolithic package | 4 modular sub-packages |
| No fine-tuning trainers | SFT, DPO, RLAIF, RLVR trainers |
