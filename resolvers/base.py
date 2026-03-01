"""Instrument resolver interface and shared data model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Instrument:
    """Represents a single option contract."""
    key:         str
    symbol:      str
    strike:      float
    option_type: str           # 'CE' or 'PE'
    expiry:      datetime
    expiry_str:  str = field(default="")   # human-readable e.g. "17 MAR 26"

    def __repr__(self) -> str:
        return f"<Instrument {self.symbol} {self.option_type} {self.strike} exp={self.expiry.date()}>"


class InstrumentResolver(ABC):
    """
    Abstract resolver — maps (symbol, spot_price, reference_date) to a list
    of option Instrument objects around the ATM strike.

    Each exchange/mode has its own concrete implementation.
    """

    @abstractmethod
    def resolve(
        self,
        symbol:         str,
        spot_price:     float,
        reference_date: str,
        num_strikes:    int = 3,
    ) -> tuple[list[Instrument], Optional[datetime], bool]:
        """
        Returns
        -------
        instruments  : list[Instrument]
        expiry_dt    : datetime | None
        is_expired   : bool
        """
