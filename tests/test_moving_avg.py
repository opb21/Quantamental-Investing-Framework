from src.strategy.moving_avg import generate_signals
import pandas as pd

def test_signal_generation():
    data = pd.DataFrame({"Close": [1, 2, 3, 4, 5]})
    result = generate_signals(data, window=2)
    assert "signal" in result.columns