import yaml
import os
from src.data_loader import StockDataLoader
from src.data_processor import TimeSeriesDataProcessor
from src.model import LSTMStockPredictor
from src.trainer import ModelTrainer
from src.visualization import TrainingVisualizer

def main():
    # 1. Load config
    config_path = "configs/train_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Data Loading
    loader = StockDataLoader(
        data_path=config['data']['filepath'],
        ticker=config['data']['ticker'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    df = loader.load_data()
    dataset = df.values

    # 3. Data Processing
    processor = TimeSeriesDataProcessor(
        seq_length=config['data']['sequence_length'],
        train_split_ratio=config['data']['train_split']
    )
    
    scaled_data = processor.fit_transform_scaler(dataset)
    split_data = processor.get_train_test_split(dataset, scaled_data)

    # 4. Model Building
    # X_train shape is (samples, seq_length, features)
    input_dim = split_data['X_train'].shape[2]
    
    lstm_model = LSTMStockPredictor(
        input_dim=input_dim,
        hidden_dim1=config['model']['lstm_units'][0],
        hidden_dim2=config['model']['lstm_units'][1],
        dense_dim=config['model']['dense_units'][0],
        output_dim=config['model']['dense_units'][1],
        dropout_rate=config['model']['dropout_rate']
    )

    # 5. Training and MLOps
    trainer = ModelTrainer(model=lstm_model, config=config)
    
    print("Starting training...")
    history_dict = trainer.train(
        X_train=split_data['X_train'], 
        y_train=split_data['y_train']
    )
    
    print("Evaluating model...")
    eval_results = trainer.evaluate(
        X_test=split_data['X_test'], 
        y_test=split_data['y_test'], 
        scaler=processor.scaler,
        processor=processor
    )
    
    trainer.save_model(export_path=config['model']['save_path'])

    # 6. Visualization
    visualizer = TrainingVisualizer(save_dir=config['output']['plot_dir'])
    visualizer.plot_learning_curve(history_dict)
    visualizer.plot_predictions(
        data=df, 
        training_data_len=split_data['training_data_len'], 
        predictions=eval_results['predictions']
    )

if __name__ == "__main__":
    main()