import pandas as pd
from .pattern_definitions import PatternDefinitions


class PatternRecognizer:
    def __init__(self, data):
        self.data = data

    def recognize_patterns(self):
        self.data['Pattern'] = None
        for i in range(2, len(self.data)):
            day = self.data.iloc[i]
            prev_day = self.data.iloc[i - 1]
            prev_prev_day = self.data.iloc[i - 2]
            days = self.data.iloc[i - 2:i + 1]

            if PatternDefinitions.is_bullish_engulfing(day, prev_day):
                self.data.at[i, 'Pattern'] = 'Bullish Engulfing'
            elif PatternDefinitions.is_bearish_engulfing(day, prev_day):
                self.data.at[i, 'Pattern'] = 'Bearish Engulfing'
            elif PatternDefinitions.is_morning_star(days):
                self.data.at[i, 'Pattern'] = 'Morning Star'
            elif PatternDefinitions.is_evening_star(days):
                self.data.at[i, 'Pattern'] = 'Evening Star'
            elif PatternDefinitions.is_hammer(day):
                self.data.at[i, 'Pattern'] = 'Hammer'
            elif PatternDefinitions.is_inverse_hammer(day):
                self.data.at[i, 'Pattern'] = 'Inverse Hammer'
            elif PatternDefinitions.is_hanging_man(day, prev_day):
                self.data.at[i, 'Pattern'] = 'Hanging Man'
            elif PatternDefinitions.is_shooting_star(day, prev_day):
                self.data.at[i, 'Pattern'] = 'Shooting Star'
            elif PatternDefinitions.is_three_white_soldiers(days):
                self.data.at[i, 'Pattern'] = 'Three White Soldiers'
            elif PatternDefinitions.is_three_black_crows(days):
                self.data.at[i, 'Pattern'] = 'Three Black Crows'
            elif PatternDefinitions.is_dark_cloud_cover(day, prev_day):
                self.data.at[i, 'Pattern'] = 'Dark Cloud Cover'

        return self.data