from typing import List, Sequence, TypedDict, Union

from darts import TimeSeries


CovariateType = Union[TimeSeries, Sequence[TimeSeries]]


class GeoJSONFeatureCollection(TypedDict):
    type: str
    features: List[dict]


class GeoJSONFeature(TypedDict):
    type: str
    bbox: List[float]
    properties: dict
    geometry: dict
