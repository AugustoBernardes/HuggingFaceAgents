from staticmap import StaticMap, CircleMarker
from smolagents import tool
from typing import TypedDict, Dict

class Coordinates(TypedDict):
    lat: float
    lng: float

Places = Dict[str, Coordinates]

@tool
def map_generation(places: Places, map_name: str) -> str:
    """
    Generate a static map image with markers for given geographic locations.

    This tool receives a dictionary of places with latitude and longitude
    coordinates and creates a PNG map with markers plotted at those locations.
    The map is centered on North America and saved locally.

    Args:
        places: A dictionary where each key is a place name and the value
                contains geographic coordinates with:
                - lat: Latitude
                - lng: Longitude
        map_name: The name of the output map file (without extension).

    Output:
        A PNG image file saved as "<map_name>.png" containing the map
        with markers representing the provided locations.

    Example input:
        {
            "Toronto": {"lat": 43.6534817, "lng": -79.3839347},
            "Vancouver": {"lat": 49.2608724, "lng": -123.113952}
        }

    Example output:
        canada_map.png
    """
    #Focusing only NA for the testing
    world_map = StaticMap(2000, 1000)
    for place in places.values():
        marker = CircleMarker((place["lng"], place["lat"]), "red", 12)
        world_map.add_marker(marker)

    image = world_map.render(
        zoom=3,
        center=(-100, 45)
    )

    image.save(f"{map_name}.png")

    return f"Map with name {map_name}.png was generated"
