import requests

url = 'http://127.0.0.1:5000/api/forecast'
payload = {
    "pollutant": "PM2.5",
    "date": "2025-05-01"
}

response = requests.post(url, json=payload)
print(response.json())
