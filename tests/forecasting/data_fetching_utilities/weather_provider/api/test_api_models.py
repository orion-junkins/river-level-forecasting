from rlf.forecasting.data_fetching_utilities.weather_provider.api.models import Response


def test_init():
    response = Response(200, 'https://fake.com', 'Fake OK',
                        {'Content-Type': 'application/json'}, [{'id': 1, 'name': 'fake name'}])
    assert response.status_code == 200
    assert response.url == 'https://fake.com'
    assert response.message == 'Fake OK'
    assert response.headers == {'Content-Type': 'application/json'}
    assert response.data == [{'id': 1, 'name': 'fake name'}]


def test_init_no_data():
    response = Response(200, 'https://fake.com', 'OK',
                        {'Content-Type': 'application/json'})
    assert response.status_code == 200
    assert response.url == 'https://fake.com'
    assert response.message == 'OK'
    assert response.headers == {'Content-Type': 'application/json'}
    assert response.data is None
