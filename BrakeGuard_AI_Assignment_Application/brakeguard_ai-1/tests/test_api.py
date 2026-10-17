from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'healthy'

def test_features():
    r = client.get('/api/v1/vehicles/BG-10245/features')
    assert r.status_code == 200
    assert 'brake_temperature' in r.json()['features']

def test_prediction():
    r = client.post('/api/v1/predictions?vehicle_id=BG-10245')
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body['fault_probability'] <= 1
    assert body['risk'] in ['LOW','MEDIUM','HIGH']
    assert body['model_version'] == 'brakeguard-rf-v1.0'

def test_event_driven_telemetry():
    payload = {'vehicle_id':'BG-55555','brake_temperature':180,'brake_pressure':70,'pad_wear_pct':75,'vibration_rms':3,'fluid_age_days':400,'odometer_km':80000}
    r = client.post('/api/v1/telemetry', json=payload)
    assert r.status_code == 200
    assert client.get('/api/v1/vehicles/BG-55555/features').status_code == 200
