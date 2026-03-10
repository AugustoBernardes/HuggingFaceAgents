import requests
import math


def distance_km(lat1, lon1, lat2, lon2):
    R = 6371

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return format(R * c, ".2f") + " Km"


def calculate_distance(city1, city2):
    url = "https://nominatim.openstreetmap.org/search"

    params1 = {
        "q": city1,
        "format": "json"
    }

    params2 = {
        "q": city2,
        "format": "json"
    }

    headers = {
        "User-Agent": "my-dummy-agent-app"
    }

    response1 = requests.get(url, params=params1, headers=headers)
    response2 = requests.get(url, params=params2, headers=headers)

    data1 = response1.json()
    data2 = response2.json()

    lat1, lon1 = float(data1[0]["lat"]), float(data1[0]["lon"])
    lat2, lon2 = float(data2[0]["lat"]), float(data2[0]["lon"])

    return distance_km(lat1, lon1, lat2, lon2)

print(calculate_distance("New York", "Los Angeles"))