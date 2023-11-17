class PatternDefinitions:
    @staticmethod
    def is_bullish(open_price, close_price):
        return close_price > open_price

    @staticmethod
    def is_bearish(open_price, close_price):
        return close_price < open_price

    @staticmethod
    def is_hammer(day):
        body = abs(day['Close'] - day['Open'])
        lower_shadow = min(day['Open'], day['Close']) - day['Low']
        return lower_shadow > 2 * body and body < (day['High'] - day['Low']) / 3

    @staticmethod
    def is_inverse_hammer(day):
        body = abs(day['Close'] - day['Open'])
        upper_shadow = day['High'] - max(day['Open'], day['Close'])
        return upper_shadow > 2 * body and body < (day['High'] - day['Low']) / 3

    @staticmethod
    def is_bullish_engulfing(day, prev_day):
        return PatternDefinitions.is_bullish(day['Open'], day['Close']) and \
               PatternDefinitions.is_bearish(prev_day['Open'], prev_day['Close']) and \
               day['Open'] < prev_day['Close'] and day['Close'] > prev_day['Open']

    @staticmethod
    def is_piercing_line(day, prev_day):
        return PatternDefinitions.is_bearish(prev_day['Open'], prev_day['Close']) and \
               PatternDefinitions.is_bullish(day['Open'], day['Close']) and \
               day['Open'] < prev_day['Close'] and day['Close'] > (prev_day['Open'] + prev_day['Close']) / 2

    @staticmethod
    def is_morning_star(days):
        if len(days) != 3:
            return False

        return PatternDefinitions.is_bearish(days.iloc[0]['Open'], days.iloc[0]['Close']) and \
            min(days.iloc[1]['Open'], days.iloc[1]['Close']) > days.iloc[0]['Close'] and \
            PatternDefinitions.is_bullish(days.iloc[2]['Open'], days.iloc[2]['Close']) and \
            days.iloc[2]['Close'] > days.iloc[1]['Close']

    @staticmethod
    def is_three_white_soldiers(days):
        if len(days) != 3:
            return False

        return all(PatternDefinitions.is_bullish(row['Open'], row['Close']) for _, row in days.iterrows()) and \
            days.iloc[0]['Close'] < days.iloc[1]['Open'] < days.iloc[1]['Close'] < days.iloc[2]['Open']

    @staticmethod
    def is_hanging_man(day, prev_day):
        return PatternDefinitions.is_hammer(day) and \
               PatternDefinitions.is_bearish(day['Open'], day['Close']) and \
               prev_day['Close'] < day['Close']

    @staticmethod
    def is_shooting_star(day, prev_day):
        return PatternDefinitions.is_inverse_hammer(day) and \
               PatternDefinitions.is_bearish(day['Open'], day['Close']) and \
               prev_day['Close'] < day['Close']

    @staticmethod
    def is_bearish_engulfing(day, prev_day):
        return PatternDefinitions.is_bullish(prev_day['Open'], prev_day['Close']) and \
               PatternDefinitions.is_bearish(day['Open'], day['Close']) and \
               day['Open'] > prev_day['Close'] and day['Close'] < prev_day['Open']

    @staticmethod
    def is_evening_star(days):
        if len(days) != 3:
            return False

        return PatternDefinitions.is_bullish(days.iloc[0]['Open'], days.iloc[0]['Close']) and \
            min(days.iloc[1]['Open'], days.iloc[1]['Close']) < days.iloc[0]['Close'] and \
            PatternDefinitions.is_bearish(days.iloc[2]['Open'], days.iloc[2]['Close']) and \
            days.iloc[2]['Close'] < days.iloc[1]['Close']

    @staticmethod
    def is_three_black_crows(days):
        if len(days) != 3:
            return False

        return all(PatternDefinitions.is_bearish(row['Open'], row['Close']) for _, row in days.iterrows()) and \
            days.iloc[0]['Open'] > days.iloc[1]['Open'] > days.iloc[2]['Open']

    @staticmethod
    def is_dark_cloud_cover(day, prev_day):
        return PatternDefinitions.is_bullish(prev_day['Open'], prev_day['Close']) and \
               PatternDefinitions.is_bearish(day['Open'], day['Close']) and \
               day['Open'] > prev_day['Close'] and \
               day['Close'] < (prev_day['Open'] + prev_day['Close']) / 2