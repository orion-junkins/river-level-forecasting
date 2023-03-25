from typing import List, Sequence, TypedDict, Union

from darts import TimeSeries


CovariateType = Union[TimeSeries, Sequence[TimeSeries]]


class GeoJSONFeatureCollection(TypedDict):
    type: str
    features: List[dict]
