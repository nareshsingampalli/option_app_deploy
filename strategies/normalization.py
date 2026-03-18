from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class NormalizationStrategy(ABC):
    """
    Open-Closed Principle: To add a new way of processing ROC data, 
    just inherit from this class and register it in the Factory.
    """
    @abstractmethod
    def process(self, series: pd.Series, **kwargs) -> pd.Series:
        pass

class DefaultROCStrategy(NormalizationStrategy):
    """Standard ROC (%) calculation with hard clipping/clamping."""
    def process(self, series, window=5, clip_range=(-100, 100), **kwargs):
        roc = (series.pct_change() * 100).replace([np.inf, -np.inf], 0).fillna(0)
        
        mask_series = kwargs.get('mask_series')
        mask_threshold = kwargs.get('mask_threshold')
        if mask_series is not None and mask_threshold is not None:
            # Drop noisy values where prev was near zero
            roc = roc.mask(mask_series.shift(1) < mask_threshold, 0)
        
        roc = roc.clip(*clip_range)
        return roc.rolling(window=window, min_periods=1).mean().round(2)

class SoftClipStrategy(NormalizationStrategy):
    """Use tanh to gently squash spikes towards +/- limit without hard clipping."""
    def process(self, series, window=5, limit=50, scale=15, **kwargs):
        roc = (series.pct_change() * 100).replace([np.inf, -np.inf], 0).fillna(0)
        
        mask_series = kwargs.get('mask_series')
        mask_threshold = kwargs.get('mask_threshold')
        if mask_series is not None and mask_threshold is not None:
            # Kill noise where base value (e.g. IV) was too low to be reliable
            roc = roc.mask(mask_series.shift(1) < mask_threshold, 0)
            
        # Smoothly squashes large values. 
        # With limit=50 and scale=15:
        # - A 15% ROC becomes ~38 (50 * tanh(1))
        # - A 1000% ROC stays close to 50.
        processed = limit * np.tanh(roc / scale)
        return processed.rolling(window=window, min_periods=1).mean().round(2)

class LogROCStrategy(NormalizationStrategy):
    """Dampens spikes using a log1p transform (retains negative direction)."""
    def process(self, series, window=5, **kwargs):
        roc = (series.pct_change() * 100).replace([np.inf, -np.inf], 0).fillna(0)
        # sign(x) * log(1 + |x|)
        processed = np.sign(roc) * np.log1p(np.abs(roc))
        return processed.rolling(window=window, min_periods=1).mean().round(2)

class AbsoluteDiffStrategy(NormalizationStrategy):
    """Uses absolute difference (Basis Points) instead of percentage."""
    def process(self, series, window=5, **kwargs):
        diff = series.diff().fillna(0)
        return diff.rolling(window=window, min_periods=1).mean().round(2)

class NormalizationFactory:
    _strategies = {
        'default': DefaultROCStrategy(),
        'soft_clip': SoftClipStrategy(),
        'log': LogROCStrategy(),
        'absolute': AbsoluteDiffStrategy()
    }
    
    @classmethod
    def get_strategy(cls, name: str) -> NormalizationStrategy:
        return cls._strategies.get(name, cls._strategies['default'])
