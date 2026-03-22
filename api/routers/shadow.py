from fastapi import APIRouter

router = APIRouter(prefix="/shadow", tags=["Shadow"])

@router.get("/")
def shadow_check():
    return {"status": "ok"}