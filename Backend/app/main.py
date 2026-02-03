from fastapi import FastAPI
from datetime import datetime
import random

app = FastAPI()

@app.get("/greet/{name}")
def greet_user(name: str):
    greetings = ["Hello", "Hi", "Hey", "Greetings", "Howdy", "Salutations"]
    greeting = random.choice(greetings)
    return {"message": f"{greeting}, {name}!"}

@app.get("/time")
def get_time():
    now = datetime.now()
    time_str = now.strftime("%I:%M:%S %p")
    return {"current_time": time_str}
