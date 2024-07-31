import os
from email.message import EmailMessage
import ssl
import smtplib
from dotenv import load_dotenv
import logging
import requests
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
import subprocess


def install_package(package):
    cmd = f'winget install {package}'
    # Run the Command Prompt command and capture the output
    process = subprocess.Popen(["cmd", "/C", cmd], stdout=subprocess.PIPE)
    result = process.communicate()[0]

    # Print the output
    res = result.decode('utf-8')
    if "Found an existing package already installed" in res:
        if "No available upgrade found" in res:
            return "Already using the latest version"
        else:
            return "Found existing package which has been upgraded"
    elif "Successfully installed" in res:
        return f"{package} Installed successfully"
    else:
        return f"{package} not found. Is it contained in the winget database?"


def conversational_response(input):
    return input

def get_current_weather(location, unit="celsius"):
    """
    Fetches the current weather for a given location.

    Args:
    location (str): The city and country, e.g., "San Francisco, USA"
    unit (str): Temperature unit, either "celsius" or "fahrenheit"

    Returns:
    str: A string describing the current weather, or an error message
    """
    logging.info(f"Getting weather for {location}")
    base_url = "https://api.open-meteo.com/v1/forecast"

    # Set up parameters for the weather API
    params = {
        "latitude": 0,
        "longitude": 0,
        "current_weather": "true",
        "temperature_unit": unit
    }

    # Set up geocoding to convert location name to coordinates
    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
    location_parts = location.split(',')
    city = location_parts[0].strip()
    country = location_parts[1].strip() if len(location_parts) > 1 else ""

    geo_params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    try:
        # First attempt to get coordinates
        logging.info(f"Fetching coordinates for {location}")
        geo_response = requests.get(geocoding_url, params=geo_params)
        geo_response.raise_for_status()
        geo_data = geo_response.json()
        logging.debug(f"Geocoding response: {geo_data}")

        # If first attempt fails, try with full location string
        if "results" not in geo_data or not geo_data["results"]:
            geo_params["name"] = location
            geo_response = requests.get(geocoding_url, params=geo_params)
            geo_response.raise_for_status()
            geo_data = geo_response.json()
            logging.debug(f"Second geocoding attempt response: {geo_data}")

        # Extract coordinates if found
        if "results" in geo_data and geo_data["results"]:
            params["latitude"] = geo_data["results"][0]["latitude"]
            params["longitude"] = geo_data["results"][0]["longitude"]
            logging.info(
                f"Coordinates found: {params['latitude']}, {params['longitude']}")
        else:
            logging.warning(f"No results found for location: {location}")
            return f"Sorry, I couldn't find the location: {location}"

        # Fetch weather data using coordinates
        logging.info("Fetching weather data")
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        weather_data = response.json()
        logging.debug(f"Weather data response: {weather_data}")

        # Extract and format weather information
        if "current_weather" in weather_data:
            current_weather = weather_data["current_weather"]
            temp = current_weather["temperature"]
            wind_speed = current_weather["windspeed"]

            result = f"The current weather in {location} is {temp}°{unit.upper()} with a wind speed of {wind_speed} km/h."
            logging.info(f"Weather result: {result}")
            return result
        else:
            logging.warning(f"No current weather data found for {location}")
            return f"Sorry, I couldn't retrieve weather data for {location}"
    except requests.exceptions.RequestException as e:
        logging.error(f"Error occurred while fetching weather data: {str(e)}")
        return f"An error occurred while fetching weather data: {str(e)}"

# Function to get a random joke


def get_random_joke():
    """
    Fetches a random joke from an API.

    Returns:
    str: A string containing a joke, or an error message
    """
    logging.info("Fetching a random joke")
    joke_url = "https://official-joke-api.appspot.com/random_joke"

    try:
        response = requests.get(joke_url)
        response.raise_for_status()
        joke_data = response.json()
        joke = f"{joke_data['setup']} - {joke_data['punchline']}"
        logging.info(f"Random joke: {joke}")
        return joke
    except requests.exceptions.RequestException as e:
        logging.error(f"Error occurred while fetching joke: {str(e)}")
        return f"An error occurred while fetching a joke: {str(e)}"


def send_mail(to : str, subject : str, body : str):
    """
    Sends an email with a specified subject and body to a specified email

    returns:
    str: A string saying the message was sent successfully, or an error message
    """
    sender = "brian.macharia.wambui@gmail.com"
    password = os.environ.get("PASSWORD")
    em = EmailMessage()
    em["From"] = sender
    em["To"] = to
    em["Subject"] = subject
    em.set_content(body)

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(sender, password=password)
            smtp.sendmail(sender, to,em.as_string())
            return "mail sent successfully"
    except Exception as e:
        logging.error(f"Error occurred while sending mail: {str(e)}")
        return f"An error occurred while sending mail: {str(e)}"