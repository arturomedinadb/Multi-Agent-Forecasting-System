# End-to-End AI Forecasting Agent - Web UI

A Flask-based web interface for running an automated 3-stage demand forecasting pipeline powered by AI agents. Upload CSV files and let the system automatically map schemas, engineer features, and train predictive models.

## 🎯 Overview

This web application orchestrates a complete end-to-end demand forecasting workflow:

1. **Stage 1: Schema Mapping** - AI agents automatically map your raw CSV files to a standardized demand forecasting schema
2. **Stage 2: Feature Engineering** - Automated feature engineering with lag, rolling, time, promotion, holiday, weather, and economic features
3. **Stage 3: Model Training** - Multi-model training with iterative improvement using XGBoost, LightGBM, CatBoost, and Random Forest

## ✨ Features

- **Multi-File Upload**: Upload multiple CSV files simultaneously
- **Intelligent Schema Mapping**: AI-powered automatic column mapping and data transformation
- **Automated Feature Engineering**: Creates lag, rolling, time-based, promotion, holiday, weather, and economic features
- **Multi-Model Training**: Trains multiple forecasting models with automatic hyperparameter tuning
- **Iterative Improvement**: AI agents evaluate models and iteratively improve them
- **Real-Time Status Monitoring**: Auto-refreshing status page with detailed progress tracking
- **Session Management**: Track multiple workflow runs with unique session IDs
- **Comprehensive Results**: View detailed results, model metrics, and output file locations

## 📋 Prerequisites

### Required

- **Python 3.10-3.12** (Python 3.13 not yet supported)
- **OpenAI API Key** - Get one from [OpenAI](https://platform.openai.com/api-keys)
- **Package Manager** - Poetry (recommended) or pip

### Recommended System Requirements

- 4GB+ RAM (8GB+ recommended for large datasets)
- 2GB+ free disk space for outputs
- Stable internet connection for API calls

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd End-To-End-AI-Forecasting-Agent
```

### Step 2: Install Dependencies

**Using Poetry (Recommended):**

   ```bash
   poetry install
poetry shell  # Activate virtual environment
   ```

   **Using pip:**

   ```bash
   pip install -e .
   ```

**Or install manually:**

   ```bash
pip install openai-agents[sqlalchemy] flask werkzeug pandas numpy python-dotenv pyyaml jinja2 sqlalchemy aiosqlite scikit-learn xgboost lightgbm catboost
```

### Step 3: Configure Environment Variables

Create a `.env` file in the project root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
FLASK_SECRET_KEY=your_secret_key_here  # Optional, for production
```

**Important:** The `OPENAI_API_KEY` is required for the AI agents to function.

### Step 4: Verify Installation

   ```bash
python -c "from agents import Agent; print('✓ Dependencies installed correctly!')"
python -c "import flask; print('✓ Flask installed correctly!')"
   ```

## 🏃 Running the Application

### Option 1: Using the Run Script (Recommended)

From the project root:

```bash
python -m src.ui.run_server
```

Or from the `src/ui` directory:

```bash
cd src/ui
python run_server.py
```

### Option 2: Using Python Directly

From the `src/ui` directory:

```bash
cd src/ui
python app.py
```

The server will start on `http://localhost:8002` (default port configured in `app.py`)

### Option 3: Using Flask CLI

From the `src/ui` directory:

```bash
cd src/ui
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=8002
```

### Changing the Port

If port 8002 is in use, you can change it by editing `app.py` line 453:

```python
app.run(debug=True, host="0.0.0.0", port=8002)  # Change 8002 to your desired port
```

## 📖 Usage Guide

### 1. Access the Web Interface

Open your browser and navigate to:
```
http://localhost:8002
```

### 2. Upload Your Data Files

- Click **"Choose Files"** and select one or more CSV files
- Files must be in CSV format with headers
- Maximum file size: 100MB per file
- Supported columns include: transaction dates, product IDs, store IDs, sales data, promotions, holidays, weather, economic indicators, etc.

### 3. Configure Settings

- **Sample Size (Row Limit)**: Number of rows to sample per file (default: 10)
  - Use smaller values (10-100) for faster testing
  - Use larger values (100-1000) for production runs
- **Output Directory** (optional): Where results will be saved
  - Leave empty for default: `output/`
  - Specify a custom path relative to project root

### 4. Start the Workflow

- Click **"🚀 Run End-to-End Workflow"** button
- You'll be redirected to a status page showing real-time progress

### 5. Monitor Progress

The status page shows:

- **Session ID**: Unique identifier for this run
- **Current Status**: `pending` → `running` → `completed`/`error`
- **Stage Progress**: 
  - Schema Mapping status
  - Feature Engineering status
  - Training status
- **Auto-refresh**: Page updates every 5 seconds automatically

### 6. View Results

Once complete, the status page displays:

- **Overall Success Status**
- **Output Directory Location**
- **Stage-by-Stage Results**:
  - Schema mapping output files
  - Feature-engineered dataset
  - Trained models and metrics
- **Best Model Information**: Model path, performance metrics
- **Error Details** (if any): Full traceback for troubleshooting

## 📁 Output Structure

After a successful run, your output directory will contain:

```
output/
├── merged_output.csv              # Stage 1: Merged and mapped data
├── engineered_features.csv         # Stage 2: Feature-engineered dataset
└── training_results/
    ├── data/                       # Processed training data
    │   ├── X_train.csv
    │   ├── X_val.csv
    │   ├── X_test.csv
    │   ├── y_train.csv
    │   ├── y_val.csv
    │   ├── y_test.csv
    │   ├── scaler.pkl
    │   ├── feature_names.json
    │   └── id_columns.json
    ├── models/                     # Trained models
    │   └── [session_uuid]/
    │       ├── model_name.pkl
    │       └── model_name_metadata.json
    └── inference/                  # Best model for inference
        └── [timestamp]/
            └── best_model.pkl
```

## 🏗️ Architecture

### Three-Stage Pipeline

```
┌─────────────────┐
│   CSV Files     │
│  (Multiple)     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Stage 1: Schema Mapping            │
│  - AI agent maps columns            │
│  - Data transformation              │
│  - Merges multiple files            │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Stage 2: Feature Engineering       │
│  - Lag features                     │
│  - Rolling statistics               │
│  - Time-based features              │
│  - Promotion/holiday/weather/       │
│    economic features                │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Stage 3: Model Training            │
│  - Multi-model training             │
│  - Iterative improvement            │
│  - Best model selection             │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Trained Models │
│  & Predictions  │
└─────────────────┘
```

### Technology Stack

- **Frontend**: Flask + Jinja2 templates + HTML/CSS
- **Backend**: Flask (Python)
- **AI Framework**: OpenAI Agents SDK
- **ML Libraries**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Data Processing**: pandas, numpy
- **Database**: SQLite (for session tracking)
- **Validation**: Pydantic

## 🔧 Configuration

### Application Settings

Edit `src/ui/app.py` to customize:

```python
# Maximum file upload size (default: 100MB)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

# Upload directory (default: uploads/)
app.config["UPLOAD_FOLDER"] = PROJECT_ROOT / "uploads"

# Flask secret key (default: from .env or dev key)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
```

### Workflow Settings

Default settings in `run_full_pipeline()`:

```python
target_column = "units_sold"  # Target variable for forecasting
id_columns = ["transaction_date", "product_id", "store_id"]  # ID columns
model_types = ["xgboost", "lightgbm", "catboost"]  # Models to train
```

### Model Training Parameters

Model hyperparameters and training settings can be adjusted in:
- `src/mapping_system/demand_forecasting/tools/training_functions.py`
- Agent instructions in `src/mapping_system/demand_forecasting/agents/demand_forecasting_agent.py`

## 🐛 Troubleshooting

### Import Errors

**Error**: `ModuleNotFoundError` or `ImportError`

**Solutions**:
1. Ensure you're in the correct directory
2. Verify dependencies are installed: `pip install -e .` or `poetry install`
3. Check that `src` directory is in Python path (handled automatically by app.py)

### File Upload Issues

**Error**: File upload fails or files rejected

**Solutions**:
- Ensure files are valid CSV format with headers
- Check file size (max 100MB per file)
- Verify write permissions for `uploads/` directory
- Check file encoding (should be UTF-8)

### API Key Errors

**Error**: `OpenAI API key not found` or authentication errors

**Solutions**:
1. Verify `.env` file exists in project root
2. Check `OPENAI_API_KEY` is set correctly
3. Ensure no extra spaces or quotes around the API key
4. Test API key: `python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Key loaded:', bool(os.getenv('OPENAI_API_KEY')))"`

### Workflow Errors

**Error**: Workflow fails at a specific stage

**Solutions**:
1. Check the error traceback in the status page
2. Verify input files are correctly formatted
3. Ensure sufficient disk space for outputs
4. Check that all required columns are present in input files
5. Review agent logs in terminal output

### Port Already in Use

**Error**: `Port 8002 is in use by another program`

**Solutions**:
1. **Find and stop the process using the port:**
   ```bash
   lsof -i :8002
   kill <PID>
   ```

2. **Or use a different port:**
   - Edit `app.py` line 453 and change the port number

### Memory Issues

**Error**: Out of memory errors during training

**Solutions**:
- Reduce `row_limit` (sample size)
- Process fewer files at once
- Close other applications
- Increase system RAM or use a machine with more memory

### Database Errors

**Error**: SQLite database locked or errors

**Solutions**:
- Ensure no other processes are accessing `training_database.db`
- Delete and recreate the database file if corrupted
- Check file permissions on database file

## 📝 Session Management

Each workflow run creates a unique session:

- **Session ID Format**: `YYYYMMDDTHHMMSSZ` (e.g., `20241114T172648Z`)
- **Status States**: 
  - `pending` - Workflow queued but not started
  - `running` - Workflow in progress
  - `completed` - All stages completed successfully
  - `error` - Workflow failed
  - `partial` - Some stages completed, some failed/pending

**Note**: Sessions are stored in-memory by default. On server restart, all sessions are cleared. For production, consider using Redis or a database-backed session store.

## 🔒 Security Considerations

### Development Mode

The application runs with `debug=True` by default, which:
- Enables detailed error pages
- Auto-reloads on code changes
- Shows Python stack traces

**Do not use in production!**

### Production Deployment

For production deployment:

1. **Set `debug=False`** in `app.py`
2. **Use a strong `FLASK_SECRET_KEY`** in `.env`
3. **Use a production WSGI server** (e.g., Gunicorn, uWSGI)
4. **Set up proper authentication** if exposing publicly
5. **Use HTTPS** for secure connections
6. **Implement rate limiting** for API endpoints
7. **Set up proper logging** and monitoring

### Example Production Configuration

```python
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8002)
```

Or use Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:8002 'src.ui.app:app'
```

## 🧪 Testing

### Manual Testing

1. Start the server
2. Upload sample CSV files from the `data/` directory
3. Monitor workflow execution in the status page
4. Verify results in the output directory

### Sample Data

Test files are available in the `data/` directory:
- `transaction_like_synth.csv` - Transaction data
- `product_like_synth_wBrand.csv` - Product information
- `store_like_synth.csv` - Store information
- Additional CSV files for testing

## 📚 Additional Resources

### Documentation

- Main README: `README.md` (project root)
- System Documentation: `DEMAND_FORECASTING_SYSTEM.md`
- Schema Documentation: `docs/demand_forecasting_schema.md`
- UI Implementation Plan: `docs/ui_implementation_plan.md`

### Code Structure

```
src/ui/
├── __init__.py
├── app.py                 # Main Flask application
├── run_server.py          # Server startup script
├── README.md              # This file
└── templates/             # Jinja2 HTML templates
    ├── base.html          # Base template with styling
    ├── index.html         # Upload form page
    └── status.html        # Status/results page
```

## 🤝 Contributing

1. Follow the existing code style
2. Add tests for new features
3. Update documentation as needed
4. Ensure all dependencies are in `pyproject.toml`

## 📄 License

MIT License - See LICENSE file in project root

## 👥 Authors

- Katherine Gong
- Ayda Elzohbi
- Arturo Medina
- Arial Huang

## 🆘 Support

For issues, questions, or contributions:
1. Check the troubleshooting section above
2. Review error messages in the status page
3. Check terminal output for detailed logs
4. Open an issue on the project repository

---

**Last Updated**: 2024-11-14
