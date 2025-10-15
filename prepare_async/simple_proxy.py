from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
import logging

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index():
    from fastapi.responses import FileResponse
    return FileResponse("chatbot.html")

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, path: str):
    """단순 프록시 - 그냥 전달만"""

    # 모든 요청을 RumAgent로 그대로 전달
    url = f"http://211.188.53.220:2025/{path}"
    print(f"프록시 요청: {request.method} {url}")

    async with httpx.AsyncClient(timeout=120) as client:
        # 스트림 요청인지 확인
        if "/stream" in path:
            # 스트림은 별도 처리
            from fastapi.responses import StreamingResponse

            async def stream_proxy():
                # 요청 헤더 정리
                headers = {
                    'Accept': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    'Last-Event-ID': '-1'
                }

                # 필요한 헤더만 복사
                if 'user-agent' in request.headers:
                    headers['User-Agent'] = request.headers['user-agent']

                print(f"스트림 요청: {url}")
                print(f"헤더: {headers}")

                async with client.stream(
                    method=request.method,
                    url=url,
                    headers=headers,
                    content=await request.body(),
                    params=request.query_params
                ) as response:
                    print(f"스트림 응답 시작, 상태: {response.status_code}")
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            print(f"청크 수신: {len(chunk)} bytes")
                            yield chunk

            return StreamingResponse(
                stream_proxy(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )
        else:
            print(f"일반 요청: {url}")
            # 일반 요청
            response = await client.request(
                method=request.method,
                url=url,
                headers=dict(request.headers),
                content=await request.body(),
                params=request.query_params
            )

            from fastapi.responses import Response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)