import pytest
from forecasting.data_fetching_utilities.api.models import Response


class TestResponse():

    def test_init(self):
        response = Response(200, 'https://fake.com', 'Fake OK',
                            {'Content-Type': 'application/json'}, [{'id': 1, 'name': 'fake name'}])
        assert response.status_code == 200
        assert response.url == 'https://fake.com'
        assert response.message == 'Fake OK'
        assert response.headers == {'Content-Type': 'application/json'}
        assert response.data == [{'id': 1, 'name': 'fake name'}]

    def test_init_no_data(self):
        response = Response(200, 'https://fake.com', 'OK',
                            {'Content-Type': 'application/json'})
        assert response.status_code == 200
        assert response.url == 'https://fake.com'
        assert response.message == 'OK'
        assert response.headers == {'Content-Type': 'application/json'}
        assert response.data == []
