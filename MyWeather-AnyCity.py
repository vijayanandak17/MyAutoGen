import os
import asyncio
import aiohttp
import nest_asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

weather_api_key = os.getenv("OPENWEATHER_API_KEY")
print(weather_api_key)

if not weather_api_key:
    raise ValueError("❌ OPENWEATHER_API_KEY not found in .env file")

# Function to get real weather from OpenWeatherMap API
async def get_weather(city: str):
    """Fetch real-time temperature and condition for a given city."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}&units=metric"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                print(f"⚠️ Could not fetch weather for {city}. Check city name or API key.")
                data = await response.text()
                print("Response:", data)
                return None
            data = await response.json()
            temp = data["main"]["temp"]
            condition = data["weather"][0]["description"].capitalize()
            print(f"🌆 City: {city.title()}")
            print(f"🌡️ Temperature: {temp}°C")
            print(f"☁️ Condition: {condition}")
            print("-" * 50)
            return temp

# Main logic
async def main():
    print("🤖 Weather Assistant is ready!")
    print("Type a city name to get the live temperature, or 'exit' to quit.")
    print("-" * 60)

    while True:
        city = input("🌆 Enter city: ").strip()
        if city.lower() in ["exit", "quit"]:
            print("👋 Goodbye! Stay sunny!")
            break
        elif not city:
            print("⚠️ Please enter a valid city name.")
            continue
        await get_weather(city)

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())
