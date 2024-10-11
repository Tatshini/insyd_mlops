import requests

# Define the URL of the FastAPI endpoint
url = "http://localhost:8000/predict"

# Define the payload for the POST request
payload = {
    "title": "Piso Carrer de llull. Piso con 4 habitaciones con ascensor y calefacción",
    "loc_string": "Barcelona - El Parc i la Llacuna del Poblenou",
    "loc": "None",
    "features": ["87 m2", "4 hab.", "1 baño"],
    "description": "Contactar con Camila 7. 3. La Casa Agency Estudio Miraflores tiene el placer de presentarles esta es...",
    "type": "FLAT",
    "subtype": "FLAT",
    "selltype": "SECOND_HAND",
    "desc": "Contactar con Camila 7. 3. La Casa Agency Estudio Miraflores tiene",
    "id": 0
}

# Send the POST request
response = requests.post(url, json=payload)

# Print the response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
