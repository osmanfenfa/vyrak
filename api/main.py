from fastapi import FastAPI
from api.routers import sessions, profiles, shadow, health

app = FastAPI(title="Zelvryn Security Analytics")

app.include_router(sessions.router)
app.include_router(profiles.router)
app.include_router(shadow.router)
app.include_router(health.router)

@app.get("/")
def root():
    return {"message": "Zelvryn"}

@app.post("/")
def root():
    return {"message": "Zelvryn"}