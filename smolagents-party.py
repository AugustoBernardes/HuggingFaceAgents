from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, tool
import time
import datetime

@tool
def party_preparation() -> str:
    """
    This tool is responsible for calculating the total preparation time for the party.

    Alfred needs to prepare for the party. Here are the tasks:
    1. Prepare the drinks - 30 minutes
    2. Decorate the mansion - 60 minutes
    3. Set up the menu - 45 minutes
    4. Prepare the music and playlist - 45 minutes
    """

    drinks = 30
    decoration = 60
    menu = 45
    music = 45

    total_time = drinks + decoration + menu + music

    return f"The total time required to prepare the party is {total_time} minutes."

@tool
def suggested_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion (str): The type of occasion for the party. Allowed values are:
                        - "casual": Menu for casual party.
                        - "formal": Menu for formal party.
                        - "superhero": Menu for superhero party.
                        - "custom": Custom menu.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."

agent = CodeAgent(tools=[suggested_menu,party_preparation], model=InferenceClientModel(), additional_authorized_imports=["datetime"])

agent.run("Prepare a meal to friends game night and how long will take to prepare it")