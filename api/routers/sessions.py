from fastapi import APIRouter

router = APIRouter(prefix="/session", tags=["Session"])

@router.post("/")
def session_check():
    return {"status": "ok"}