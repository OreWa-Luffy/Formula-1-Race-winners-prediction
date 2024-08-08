import pandas as pd
from meteostat import Point, Daily
import time


# Fetches weather data from a list of locations, uses meteostat to fetch the historical weather for that day.
def fetch_weather_data(lat_lon, date):
    start = pd.to_datetime(date, format='%Y-%m-%d')
    end = start
    point = Point(lat_lon[0], lat_lon[1])
    data = Daily(point, start, end)
    data = data.fetch()
    if not data.empty:
        return data.iloc[0]
    else:
        return None


# Uses a list of grand prix locations to store the lat and lon coords for later use
def get_lat_lon(location):
    location_coords = {
        "Sakhir Bahrain": (26.0325, 50.5106),
        "Jeddah Saudi Arabia": (21.4858, 39.1925),
        "Melbourne Australia": (-37.8497, 144.968),
        "Imola Italy": (44.3439, 11.7167),
        "Miami USA": (25.958, -80.2389),
        "Barcelona Spain": (41.57, 2.2611),
        "Monte Carlo Monaco": (43.7347, 7.4206),
        "Baku Azerbaijan": (40.3725, 49.8533),
        "Montreal Canada": (45.5048, -73.5262),
        "Silverstone UK": (52.0786, -1.0169),
        "Spielberg Austria": (47.2197, 14.7647),
        "Le Castellet France": (43.2508, 5.7917),
        "Budapest Hungary": (47.5789, 19.2486),
        "Lusail Qatar": (25.4185, 51.5008),
        "Shanghai China": (31.2304, 121.4737),
        "Zandvoort Netherlands": (52.3889, 4.5408),
        "Monza Italy": (45.6156, 9.2811),
        "Singapore": (1.2914, 103.863),
        "Suzuka Japan": (34.8431, 136.5417),
        "Austin USA": (30.1328, -97.6411),
        "Mexico City Mexico": (19.4042, -99.0907),
        "São Paulo Brazil": (-23.7036, -46.6997),
        "Yas Marina UAE": (24.467, 54.6031),
        "Spa, Belgium": (50.4923, 5.8623),
        "Las Vegas USA": (36.1716, 115.1391)

    }
    return location_coords.get(location, None)


def main():
    # Date and location of races
    weatherFilePath = "archive/Weather.csv"
    race_schedule = pd.read_csv(weatherFilePath)
    race_schedule['date'] = pd.to_datetime(race_schedule['date'], format='%Y-%m-%d')

    weather_date = []

    # Goes through the different locations
    for _, race in race_schedule.iterrows():
        location = race["location"].strip()
        date = race["date"].strftime('%Y-%m-%d')
        lat_lon = get_lat_lon(location)
        if lat_lon:
            data = fetch_weather_data(lat_lon, date)
            if data is not None:
                # Creates a database with location, date, temp, wind and precip.
                weather_info = {
                    "location": location,
                    "Date": date,
                    "Temperature": data['tavg'],
                    "Wind": data['wspd'],
                    "Precipitation": data['prcp']
                }
                weather_date.append(weather_info)

            # Prevents the system from overloading. Does slow down process though
            time.sleep(1)

    # Fetches the data into a csv file.
    weather_df = pd.DataFrame(weather_date)

    # Fills in places that have no precip with a 0.
    weather_df['Precipitation'].fillna(0, inplace=True)

    # Outputs the data to a csv.
    output_path = "archive/weather_data.csv"
    weather_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
