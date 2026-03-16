import os
import asyncio
import aiohttp
import ssl
import certifi
from typing import Dict, List
from smolagents import tool
from dotenv import load_dotenv

load_dotenv()

GEO_API_KEY = os.getenv("GEO_API_KEY")

ssl_context = ssl.create_default_context(cafile=certifi.where())


async def fetch(session, city: str) -> Dict:
    url = "https://api.opencagedata.com/geocode/v1/json"

    params = {
        "q": city,
        "key": GEO_API_KEY,
        "limit": 5,
        "no_annotations": 1
    }

    async with session.get(url, params=params) as resp:
        geo_search_response = await resp.json()

        if not geo_search_response["results"]:
            return {"city": city, "lat": None, "lng": None}

        possible_city_locations = geo_search_response["results"]

        geometry = None

        for location in possible_city_locations:
            if location["components"].get("_type") == "city":
                geometry = location["geometry"]
                break

        if geometry is None:
            return {"city": city, "lat": None, "lng": None}

        return {
            "city": city,
            "lat": geometry["lat"],
            "lng": geometry["lng"]
        }


async def main(cities: List[str]):
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=ssl_context)
    ) as session:
        tasks = [fetch(session, city) for city in cities]
        return await asyncio.gather(*tasks)

@tool
def cities_coordinates(cities: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Retrieve geographic coordinates for a list of cities.

    Args:
        cities: List of city names.

    Returns:
        Dictionary mapping each city to its latitude and longitude.

    Example:
        {
            "Toronto": {"lat": 43.6534817, "lng": -79.3839347},
            "Vancouver": {"lat": 49.2608724, "lng": -123.113952}
        }
    """
    results = asyncio.run(main(cities))

    return {
        item["city"]: {"lat": item["lat"], "lng": item["lng"]}
        for item in results
    }
