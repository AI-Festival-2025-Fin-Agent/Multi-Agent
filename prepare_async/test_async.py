import asyncio
import httpx
from pydantic import BaseModel

class AutoSearchRequest(BaseModel):
    query: str

async def test_async(query):
    url = "http://localhost:10000/auto-search"
    request = AutoSearchRequest(query=query)
    
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(
            url,
            json=request.model_dump()
        )
        print(response.status_code)
        print(response.json())

if __name__ == "__main__":
    asyncio.run(test_async("미래에셋증권 자사주 소각 진짜야?"))
