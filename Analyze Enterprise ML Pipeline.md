---
tags: [skill, hermes]
category: ⚙️ MLOps
skill_id: analyze-enterprise-ml-pipeline
---

# Analyze Enterprise ML Pipeline

**分类:** ⚙️ MLOps
**Skill ID:** `analyze-enterprise-ml-pipeline`

> Systematic approach to reverse-engineering and documenting complex enterprise ML pipelines, data flows, and layered prediction systems

---


# Analyze Enterprise ML Pipeline Architecture

Use this skill when you need to understand, document, or reverse-engineer complex enterprise ML pipelines, especially those involving Databricks, MLflow, layered predictions, and multi-stage data processing.

## When to Use

- Analyzing existing ML pipelines you didn't build
- Documenting data flows for compliance or knowledge transfer
- Understanding layered prediction architectures
- Reverse-engineering Databricks/Azure Data Factory workflows
- Troubleshooting complex ML system interactions
- Planning migrations or refactoring of legacy ML systems

## Methodology

### Phase 1: Initial Exploration
1. **File system scan**: Use `search_files(target='files')` to get complete file listing
2. **Pattern identification**: Look for common patterns:
   - `*ingestion*`, `*processing*`, `*model*`, `*deploy*` files
   - Notebook files (`.ipynb`, `.py` with Databricks commands)
   - SQL files and configuration files
3. **Key file identification**: Prioritize files with names suggesting:
   - Workflow orchestration (`trigger`, `start`, `run`)
   - Data flow (`ingestion`, `processing`, `transform`)
   - Model lifecycle (`train`, `predict`, `deploy`, `monitor`)

### Phase 2: Data Flow Analysis
1. **Start with entry points**: Look for files that initiate the pipeline
   - Files with `start`, `trigger`, `main`, `run` in name
   - Files that set MLflow experiments or run IDs
   - Files that define prediction dates or parameters

2. **Trace data dependencies**:
   - Read SQL queries to understand data sources
   - Look for table/feature store references
   - Identify input/output tables between stages
   - Map data transformations at each stage

3. **Identify processing stages**:
   - Data ingestion (from databases, APIs, files)
   - Data processing/cleaning/feature engineering
   - Model training/prediction
   - Result storage and serving
   - Monitoring and logging

### Phase 3: Architecture Reconstruction
1. **Layer identification**:
   - Look for `layer1`, `layer2`, `stage1`, `stage2` patterns
   - Identify if system uses hierarchical predictions
   - Map dependencies between layers

2. **Technology stack analysis**:
   - Databricks-specific patterns (`dbutils`, `%sql`, `%fs`)
   - MLflow usage (`mlflow.start_run`, `mlflow.log_*`)
   - Feature Store usage (`feature_store.FeatureStoreClient`)
   - Model frameworks (Prophet, scikit-learn, PyTorch, etc.)

3. **Business logic extraction**:
   - Analyze SQL queries for business rules
   - Look for mapping tables and transformation logic
   - Identify key business entities (BU, plant_code, SKU categories)

### Phase 4: Documentation
1. **Create data flow diagram**:
   ```
   Data Sources → Ingestion → Processing → Model Training → Prediction → Results
   ```

2. **Document key components**:
   - Input data sources and formats
   - Processing stages and transformations
   - Model architectures and parameters
   - Output destinations and formats
   - Scheduling and triggers

3. **Identify dependencies**:
   - Data dependencies between stages
   - Parameter passing mechanisms
   - Error handling and monitoring

## Common Patterns in Enterprise ML Systems

### Databricks/MLflow Patterns
- `mlflow.set_experiment()` - Experiment configuration
- `mlflow.start_run()` - Run tracking
- `dbutils.jobs.taskValues` - Parameter passing between notebooks
- `spark.sql()` - Data extraction from data lake
- Feature Store tables for intermediate results

### Layered Prediction Systems
- **Layer 1**: Time series models (Prophet, ARIMA) for component predictions
- **Layer 2**: Ensemble/ML models (Random Forest, XGBoost) for final predictions
- Feature engineering between layers (lag features, encodings)

### Data Flow Patterns
- Delta tables for intermediate storage
- Feature Store for reusable features
- Parameter tables for configuration
- Run ID tracking for reproducibility

## Example Commands

```python
# 1. Get file overview
search_files(path="/path/to/pipeline", target="files", limit=100)

# 2. Read key orchestration files
read_file("/path/to/pipeline/Start_New_Run.py")
read_file("/path/to/pipeline/Trigger.py")

# 3. Analyze data ingestion
read_file("/path/to/pipeline/Data_Ingestion_Time.py")

# 4. Trace processing flow
read_file("/path/to/pipeline/Data_Processing_Layer1.py")
read_file("/path/to/pipeline/Model_Layer1.py")
read_file("/path/to/pipeline/Data_Processing_Layer2.py")

# 5. Search for patterns
search_files(pattern="mlflow\.start_run", target="content")
search_files(pattern="spark\.sql", target="content")
search_files(pattern="feature_store", target="content")
```

## Pitfalls to Avoid

1. **Assuming linear flow**: Enterprise pipelines often have parallel branches
2. **Missing parameter passing**: Check `dbutils.jobs.taskValues` and widget values
3. **Overlooking configuration**: Look for YAML/JSON config files
4. **Ignoring monitoring**: Check for logging, metrics, and alerting code
5. **Forgetting business context**: SQL queries often contain key business logic

## Verification Steps

1. **Run ID traceability**: Can you trace a prediction back to its source data?
2. **Data lineage**: Can you map raw data → features → predictions?
3. **Parameter consistency**: Are parameters consistently passed between stages?
4. **Error handling**: Is there proper error handling and logging?
5. **Performance tracking**: Are metrics properly logged to MLflow?

## Output Template

When documenting an analyzed pipeline, include:

1. **Executive Summary**: Brief overview of pipeline purpose
2. **Architecture Diagram**: Visual representation of data flow
3. **Component Details**: Each stage with inputs, processing, outputs
4. **Data Dependencies**: Table/feature dependencies between stages
5. **Model Details**: Algorithms, parameters, training process
6. **Orchestration**: Scheduling, triggers, error handling
7. **Monitoring**: Logging, metrics, alerting
8. **Key Findings**: Issues, bottlenecks, improvement opportunities

This systematic approach ensures comprehensive understanding of complex ML systems, enabling effective documentation, troubleshooting, and improvement planning.
