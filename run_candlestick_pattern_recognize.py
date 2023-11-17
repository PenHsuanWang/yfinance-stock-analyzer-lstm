import pandas as pd
from src.analyzer.pattern_recognizer import PatternRecognizer

def main():
    data = pd.read_csv('./data/AAPL.csv')
    recognizer = PatternRecognizer(data)
    labeled_data = recognizer.recognize_patterns()
    labeled_data.to_csv('./data/patterned/labeled_AAPL.csv', index=False)
    print(labeled_data[['Pattern']].dropna())

if __name__ == "__main__":
    main()

