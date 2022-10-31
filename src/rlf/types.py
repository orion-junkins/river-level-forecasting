from typing import Sequence

from darts import TimeSeries


CovariateType = TimeSeries | Sequence[TimeSeries] | None
