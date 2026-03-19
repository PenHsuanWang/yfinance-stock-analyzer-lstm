# From Jupyter Notebooks to Production: An OOP Guide for Data Scientists 🚀

Welcome! If you are a Data Scientist who is used to writing linear, cell-by-cell code in Jupyter Notebooks, this repository is designed especially for you. 

This project takes a traditional **Stock Price Prediction** notebook (using `yfinance`, `pandas`, and `keras` LSTMs) and transforms it into a production-ready, **Object-Oriented Programming (OOP)** framework. 

This guide will explain *why* we made this shift, *how* your familiar notebook concepts map to this new structure, and how it sets you up for advanced MLOps practices.

---

## 🤔 Why Move Away From Notebooks?

Jupyter Notebooks are fantastic for exploration, data visualization, and quick prototyping. However, as projects grow and need to be deployed to production, notebooks introduce several challenges:
1. **Hidden State**: Running cells out of order can lead to unexpected behaviors. Variables are global.
2. **Hardcoding**: Hyperparameters (like `epochs=50` or `batch_size=32`) are often buried deep within cell code.
3. **Reusability**: If you want to use the same preprocessing logic for a different stock, you often have to copy-paste the whole notebook.
4. **Collaboration**: Git version control with `.ipynb` files is notoriously difficult to merge.

**The Solution: Object-Oriented Programming (OOP) & Separation of Concerns.**
By breaking our code into distinct, single-purpose "Classes" (Objects), we make the code modular, testable, and reusable.

---

## 🗺️ Mapping Notebook Cells to OOP Modules

Let's look at how your familiar notebook workflow translates into our new Python module structure.

### 1. Data Ingestion
* **In Notebook:** `df = yf.download('AAPL', ...)`
* **In OOP (`src/data_loader.py`):** We created the `StockDataLoader` class. Its sole job is to fetch data. If you decide to switch from Yahoo Finance to a SQL database tomorrow, you *only* change this file. The rest of the project doesn't care *how* the data was loaded, as long as it gets a DataFrame.

### 2. Feature Engineering & Scaling
* **In Notebook:** `scaler = MinMaxScaler(...)`, `X.append(...)` in a `for` loop.
* **In OOP (`src/data_processor.py`):** The `TimeSeriesDataProcessor` class encapsulates all data manipulation. It holds the `scaler` state internally, making it incredibly easy to call `inverse_transform` later without worrying about global variables.

### 3. Model Definition
* **In Notebook:** `model = Sequential()`, `model.add(LSTM(...))`
* **In OOP (`src/model.py`):** The `LSTMStockPredictor` class defines the network. Notice that it doesn't train the model; it just builds the architecture. This means you can easily create a `TransformerStockPredictor` class later and swap them out seamlessly.

### 4. The Training Loop
* **In Notebook:** `model.fit(X_train, y_train, epochs=...)`
* **In OOP (`src/trainer.py`):** The `ModelTrainer` class takes a compiled model and data, and executes the training. Because this logic is isolated, this is the perfect place to inject **MLOps tools like MLflow** to automatically track your experiments, loss curves, and model weights without cluttering your modeling code.

### 5. Plotting and Evaluation
* **In Notebook:** `plt.plot(...)`, `plt.show()`
* **In OOP (`src/visualization.py`):** The `TrainingVisualizer` class handles saving plots to disk. In a production pipeline (running on a cloud server), you don't have a screen for `plt.show()`, so this class ensures your graphs are safely saved as `.png` files in the `outputs/` folder.

---

## 🎛️ The Magic of Configuration (`configs/train_config.yaml`)

In a notebook, you might define `SEQ_LEN = 60` in Cell 3, and `EPOCHS = 100` in Cell 45. 

In this framework, **there are no hardcoded parameters in the Python code**. Everything is controlled by `train_config.yaml`. 

Want to see if a sequence length of 90 works better? Or want to predict Google (`GOOG`) instead of Apple (`AAPL`)? Just change the YAML file. You don't need to touch the Python code at all.

```yaml
data:
  ticker: "AAPL"              # Change this to test other stocks
  sequence_length: 60         
  train_split: 0.95           

model:
  lstm_units: [128, 64]       # Easily change network size
  dropout_rate: 0.0           
```

---

## 🎼 The Orchestrator: `serve_model_training.py`

If the classes in `src/` are the musicians, `serve_model_training.py` is the conductor. It reads the YAML config, instantiates the classes, and passes data between them. 

Because we abstracted the complex logic into classes, the main script reads like plain English:
1. Load Config
2. Load Data
3. Process Data
4. Build Model
5. Train Model
6. Visualize Results

---

## 🚀 Quick Start: Run it Yourself!

Ready to try the OOP approach? 

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the Pipeline**
Instead of clicking "Run All" in a notebook, you execute the orchestrator script:
```bash
python serve_model_training.py
```

**3. Check the Results**
- Your trained model `.h5` file and your prediction `.png` charts will automatically appear in the `outputs/` folder.
- **Bonus:** We included MLflow! Run `mlflow ui` in your terminal and open `http://localhost:5000` in your browser to see a professional experiment tracking dashboard for your run.

---

## 🎉 Conclusion

By adopting OOP, you transition from writing "scripts" to building "systems." 
This structure allows you to easily collaborate with Data Engineers and MLOps professionals, scale your models to the cloud, and maintain your sanity as your projects grow in complexity. Happy coding!