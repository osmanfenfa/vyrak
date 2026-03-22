from fastapi import APIRouter

router = APIRouter(prefix="/profiiles", tags=["Profile"])

@router.get("/")
def profile_check():
    return {"status": "ok"}