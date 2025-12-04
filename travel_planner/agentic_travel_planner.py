import os
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict
import logging
import sys

from crewai import Agent, Task, Crew, Process
from amadeus import Client, ResponseError

# Amadeus client - uses AMADEUS_CLIENT_ID / SECRET from system env vars
amadeus = Client()

def setup_logging():
    """Configures logging to file and console."""
    log_file = 'travel_log.log'
    
    # 1. Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if setup is called multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # Define the formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 2. File Handler (for log file)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # 3. Stream Handler (for console output)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # 4. Add handlers to the root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    
    # Log a separator for clarity in the file
    logging.info("="*50)
    logging.info("NEW TRAVEL PLANNING SESSION STARTED")
    logging.info("="*50)


# ---------- Simple domain models (non-LLM "tools") ----------
@dataclass
class FlightOption:
    airline: str
    price: str
    departure_time: str
    arrival_time: str

@dataclass
class HotelOption:
    name: str
    nightly_price: str
    address: str

@dataclass
class Activity:
    name: str
    description: str

@dataclass
class UserTripRequest:
    origin: str
    destination: str
    departure_date: str
    return_date: str

# ---------- City/Airport code lookup with caching ----------
_code_cache: Dict[str, str] = {}
_coordinate_cache: Dict[str, tuple] = {} # Cache for (latitude, longitude)

def get_city_code(city_name: str) -> str:
    """Convert city name to IATA city code with caching."""
    if city_name.upper() in _code_cache:
        return _code_cache[city_name.upper()]
    
    try:
        time.sleep(0.5)
        # Include 'view=FULL' to get coordinates for later use
        response = amadeus.reference_data.locations.get(
            keyword=city_name,
            subType="CITY",
            view="FULL", # Requesting FULL view to get coordinates
        )
        if response.data:
            code = response.data[0]["iataCode"]
            _code_cache[city_name.upper()] = code

            # Cache coordinates as well
            geo = response.data[0].get("geoCode", {})
            lat = geo.get("latitude")
            lon = geo.get("longitude")
            if lat and lon:
                _coordinate_cache[code] = (lat, lon)

            return code
        else:
            logging.info(f"[get_city_code] No city found for '{city_name}'")
            return city_name[:3].upper()
    except Exception as error:
        logging.error(f"[get_city_code] Error for '{city_name}': {error}")
        return city_name[:3].upper()

def get_airport_code(city_name: str) -> str:
    """Convert city name to primary airport code."""
    if city_name.upper() in _code_cache:
        return _code_cache[city_name.upper()]
    
    try:
        time.sleep(0.5)
        # FIX: Removed the unsupported 'max=1' parameter
        response = amadeus.reference_data.locations.get(
            keyword=city_name,
            subType="AIRPORT",
        )
        if response.data:
            code = response.data[0]["iataCode"]
            _code_cache[city_name.upper()] = code
            return code
        else:
            logging.info(f"[get_airport_code] No airport, using city code for '{city_name}'")
            return get_city_code(city_name)
    except Exception as error:
        logging.error(f"[get_airport_code] Error for '{city_name}': {error}")
        return get_city_code(city_name)
    
def get_city_coordinates(city_code: str) -> tuple | None:
    """Retrieves (latitude, longitude) for a cached city code."""
    # Check if coordinates are already cached by the IATA code
    if city_code in _coordinate_cache:
        return _coordinate_cache[city_code]
    
    # If not cached, try to fetch the location data again
    try:
        time.sleep(0.5)
        response = amadeus.reference_data.locations.get(
            keyword=city_code,
            subType="CITY",
            view="FULL",
        )
        if response.data:
            geo = response.data[0].get("geoCode", {})
            lat = geo.get("latitude")
            lon = geo.get("longitude")
            if lat and lon:
                _coordinate_cache[city_code] = (lat, lon)
                return lat, lon
        return None
    except Exception as error:
        logging.error(f"[get_city_coordinates] Error fetching coordinates for '{city_code}': {error}")
        return None

# ---------- Amadeus API calls ----------
def find_best_flight(origin: str, destination: str, dep: str, ret: str) -> FlightOption:
    """Real Amadeus Flight Offers Search API (requires airport codes)."""
    # Replaced '→' with '->' to avoid the UnicodeEncodeError in console logging
    logging.info(f"[find_best_flight] Searching {origin} (Airport) -> {destination} (Airport)")
    try:
        time.sleep(0.5)
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=dep,
            returnDate=ret,
            adults=1,
            currencyCode="USD",
            max=3,
        )
        offers = response.data
        if not offers:
            return FlightOption(
                airline="No flights found",
                price="N/A",
                departure_time="N/A",
                arrival_time="N/A",
            )

        best = offers[0]
        price = best["price"]["total"]
        first_itinerary = best["itineraries"][0]
        first_segment = first_itinerary["segments"][0]
        last_segment = first_itinerary["segments"][-1]

        airline = first_segment["carrierCode"]
        departure_time = first_segment["departure"]["at"]
        arrival_time = last_segment["arrival"]["at"]
        
        logging.info(f"[find_best_flight] Found flight with {airline} at {price} USD.")

        return FlightOption(
            airline=airline,
            price=f"{price} USD",
            departure_time=departure_time,
            arrival_time=arrival_time,
        )
    except ResponseError as error:
        logging.error(f"[find_best_flight] Amadeus error: {error}")
        return FlightOption(
            airline="Error fetching flights",
            price="N/A",
            departure_time="N/A",
            arrival_time="N/A",
        )

def find_best_hotel(destination: str, dep: str, ret: str) -> HotelOption:
    """
    Finds the best hotel offer by City Code. 
    (Revised to use amadeus.get() for direct endpoint access, bypassing the structured object that caused the 'no attribute' error.)
    """
    logging.info(f"[find_best_hotel] Searching hotels in {destination} (City Code)")
    
    try:
        # STEP 1: Get hotel offers directly by City Code (LON, PAR, NYC, etc.)
        time.sleep(0.5)
        # FIX: Use amadeus.get() with the V3 endpoint path to avoid the 'no attribute' error
        response = amadeus.get(
            '/v3/shopping/hotel-offers',
            cityCode=destination,
            checkInDate=dep,
            checkOutDate=ret,
            adults=1,
            currency="USD",
            view="FULL", # Request full details
            bestRateOnly=True # Only get the best available rate
        )
        
        # Check if the response contains data and offers
        if not response.data or not response.data[0].get("offers"):
            logging.info(f"[find_best_hotel] No offers found for hotels in {destination}.")
            return HotelOption(
                name="No hotel offers found",
                nightly_price="N/A",
                address="N/A",
            )

        # Extract the best offer from the first hotel returned
        first_hotel_offers = response.data[0]
        hotel_info = first_hotel_offers.get("hotel", {})
        offer_info = first_hotel_offers.get("offers", [{}])[0]
        
        name = hotel_info.get("name", "Unknown Hotel")
        address_lines = hotel_info.get("address", {}).get("lines", [])
        address = address_lines[0] if address_lines else "Address not available"
        price = offer_info.get("price", {}).get("total", "N/A")

        logging.info(f"[find_best_hotel] Found offer for {name}")
        
        return HotelOption(
            name=name,
            nightly_price=f"{price} USD",
            address=address,
        )
        
    except ResponseError as error:
        logging.error(f"[find_best_hotel] Amadeus API error: {error}")
        return HotelOption(
            name="Error fetching hotels (API issue)",
            nightly_price="N/A",
            address="N/A",
        )
    except Exception as error:
        logging.error(f"[find_best_hotel] General error: {error}")
        # This will now catch genuine API issues or data parsing issues
        return HotelOption(
            name="Error fetching hotels (General)",
            nightly_price="N/A",
            address="N/A",
        )


def get_activities(destination: str) -> List[Activity]:
    """Dynamically fetches activities using the Amadeus Activities API (requires coordinates)."""
    
    coordinates = get_city_coordinates(destination)
    if not coordinates:
        logging.info(f"[get_activities] Failed to get coordinates for city code {destination}.")
        return [Activity("Coordinate Error", "Cannot search activities without coordinates.")]

    latitude, longitude = coordinates
    logging.info(f"[get_activities] Searching activities near {destination} ({latitude}, {longitude})")

    try:
        # Use Amadeus Activities API v1, which requires latitude and longitude
        time.sleep(0.5)
        response = amadeus.shopping.activities.get(
            latitude=latitude,
            longitude=longitude,
            radius=20, # Increased search radius for better results
        )
        
        activities_list = []
        # Get up to 3 activities
        for item in response.data[:3]:
            name = item.get("name", "Unknown Activity")
            description = f"Category: {item.get('category', 'N/A')}"
            
            # Use short description if available
            if item.get('shortDescription'):
                description = item['shortDescription']

            # If a price is available, add it
            price_info = item.get('price', {})
            if price_info.get('currencyCode') and price_info.get('amount'):
                 description += f" (Price: {price_info['amount']} {price_info['currencyCode']})"

            activities_list.append(Activity(name, description))

        if activities_list:
            logging.info(f"[get_activities] Found {len(activities_list)} activities.")
            return activities_list
        
        # Fallback if API returns nothing
        logging.info(f"[get_activities] No dynamic activities found for {destination}, using static fallback.")
        return [Activity("Local Exploration", "Explore nearby attractions and sights."), Activity("Dining Experience", "Try a highly-rated local restaurant.")]

    except ResponseError as error:
        logging.error(f"[get_activities] Amadeus API error: {error}")
        # Fallback to static if API call fails
        return [Activity("City Walking Tour", "Explore the main sights on foot (API Error Fallback)."), Activity("Local Market", "Enjoy local food and crafts (API Error Fallback).")]
    except Exception as error:
        logging.error(f"[get_activities] General error: {error}")
        # General error fallback
        return [Activity("Activity Search Failed", "Check local listings (General Error Fallback).")]

# ---------- CLI input ----------
def get_user_trip_from_cli() -> UserTripRequest:
    print("=== Agentic Personal Travel Planner (CrewAI + LLM) ===")
    origin = input("Enter origin city name (e.g., New York): ").strip().title()
    destination = input("Enter destination city name (e.g., Paris): ").strip().title()
    dep = input("Enter departure date (YYYY-MM-DD): ").strip()
    ret = input("Enter return date (YYYY-MM-DD): ").strip()

    for d in [dep, ret]:
        try:
            datetime.strptime(d, "%Y-%m-%d")
        except ValueError:
            logging.error("Invalid date format entered.")
            raise SystemExit("Invalid date format. Use YYYY-MM-DD.")

    # Convert city names to IATA codes
    # Origin needs the specific airport code for flight search
    origin_code = get_airport_code(origin)
    # The destination for hotels/activities needs the City Code
    destination_city_code = get_city_code(destination)
    
    logging.info(f"Using origin airport: {origin_code}")
    logging.info(f"Using destination city code for hotel/activities: {destination_city_code}")

    return UserTripRequest(
        origin=origin_code,           # Airport Code (e.g., JFK)
        destination=destination_city_code, # City Code (e.g., PAR)
        departure_date=dep,
        return_date=ret,
    )

# ---------- CrewAI multi-agent setup ----------
def build_travel_planner_crew(trip: UserTripRequest) -> Crew:
    # Agent 1: Planner
    planner = Agent(
        role="Travel planner",
        goal="Understand the user's trip details and plan what information is needed.",
        backstory=(
            "You are a helpful travel planning assistant. "
            "You break down the planning task into flights, hotels, and activities."
        ),
        verbose=True,
    )

    # Agent 2: Flight researcher
    flight_agent = Agent(
        role="Flight search specialist",
        goal="Summarize the best flight option clearly.",
        backstory=(
            "You specialize in finding good flight deals using Amadeus API data."
        ),
        verbose=True,
    )

    # Agent 3: Hotel researcher
    hotel_agent = Agent(
        role="Hotel search specialist",
        goal="Summarize the best hotel option clearly.",
        backstory=(
            "You specialize in suggesting hotels using Amadeus API data."
        ),
        verbose=True,
    )

    # Agent 4: Itinerary writer
    itinerary_agent = Agent(
        role="Itinerary writer",
        goal="Write a small, clear travel itinerary.",
        backstory=(
            "You turn structured information into a concise, itemized itinerary "
            "easy to read in a terminal."
        ),
        verbose=True,
    )

    # The trip object contains:
    # trip.origin: Airport Code (e.g., JFK)
    # trip.destination: City Code (e.g., PAR)

    # Convert the destination City Code (trip.destination, e.g., PAR) into a 
    # specific Airport Code (e.g., CDG or ORY) for the flight search.
    dest_airport_code = get_airport_code(trip.destination) 

    # Get real data from APIs
    
    # 1. FLIGHT SEARCH: MUST use Airport Codes for both origin (trip.origin) 
    #    and destination (dest_airport_code).
    flight = find_best_flight(trip.origin, dest_airport_code, trip.departure_date, trip.return_date)
    
    # 2. HOTEL & ACTIVITIES: MUST use the City Code (which is stored in trip.destination).
    hotel = find_best_hotel(trip.destination, trip.departure_date, trip.return_date)
    # Activities now fetches real-time data using coordinates derived from the City Code
    activities = get_activities(trip.destination)

    activities_text = "\n".join([f"- {a.name}: {a.description}" for a in activities])

    # Tasks
    planner_task = Task(
        description=(
            f"User wants to travel from {trip.origin} to {trip.destination} "
            f"({trip.departure_date} to {trip.return_date}). "
            "Plan what information is needed and delegate to specialists."
        ),
        agent=planner,
        expected_output="Short plan of what each specialist should provide.",
    )

    flight_task = Task(
        description=(
            f"Flight data:\n"
            f"- Airline: {flight.airline}\n"
            f"- Price: {flight.price}\n"
            f"- Departure: {flight.departure_time}\n"
            f"- Arrival: {flight.arrival_time}\n\n"
            "Summarize as 2-3 bullet points for itinerary."
        ),
        agent=flight_agent,
        expected_output="2-3 bullet points describing the flight.",
    )

    hotel_task = Task(
        description=(
            f"Hotel data:\n"
            f"- Name: {hotel.name}\n"
            f"- Price: {hotel.nightly_price}\n"
            f"- Address: {hotel.address}\n\n"
            "Summarize as 2-3 bullet points for itinerary."
        ),
        agent=hotel_agent,
        expected_output="2-3 bullet points describing the hotel.",
    )

    itinerary_task = Task(
        description=(
            # Replaced '→' with '->' here to prevent UnicodeEncodeError in verbose logging
            f"Trip: {trip.origin} -> {trip.destination} ({trip.departure_date} to {trip.return_date})\n"
            f"Activities:\n{activities_text}\n\n"
            "Use flight/hotel summaries above. Create SMALL itinerary:\n"
            "1. Flight\n2. Hotel\n3. Activities\n4. Simple day plan (3 lines max)"
        ),
        agent=itinerary_agent,
        expected_output="Compact travel itinerary.",
    )

    return Crew(
        agents=[planner, flight_agent, hotel_agent, itinerary_agent],
        tasks=[planner_task, flight_task, hotel_task, itinerary_task],
        process=Process.sequential,
        verbose=True,
    )

def main():
    setup_logging()
    
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY not found. Set as system variable.")
        # Raise SystemExit after logging the error
        raise SystemExit("OPENAI_API_KEY not found. Set as system variable.")

    trip = get_user_trip_from_cli()
    crew = build_travel_planner_crew(trip)
    logging.info("\nRunning multi-agent crew...\n")
    result = crew.kickoff()

    # 1. Log the result (captures in file with timestamp/level, and in console with formatting)
    logging.info("\n=== Final Itinerary (LOGGED) ===")
    logging.info(result)
    
    # 2. Print the result clean to the console (for better user readability, without logging prefixes)
    print("\n\n=== Final Itinerary (PRINTED) ===")
    print(result)
    
if __name__ == "__main__":
    main()