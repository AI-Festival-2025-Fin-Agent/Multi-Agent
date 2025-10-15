import asyncio
import httpx
from pydantic import BaseModel

PUBV2AGENT_URL = "http://localhost:6200"

class AutoSearchRequest(BaseModel):
    query: str

async def run_pub_v2_agent_stream(query: str):
    """Run 생성 후 Join Run Stream으로 실시간 스트리밍"""
    request_data = {
        "assistant_id": "summarize",
        "input": {"query": query},
        "if_not_exists": "create",
        "stream_resumable": True  # 스트림 재개 가능하도록 설정
    }

    async with httpx.AsyncClient(timeout=300) as client:
        # 1️⃣ /runs로 요청 보내기 → run_id와 thread_id 받기
        response = await client.post(f"{PUBV2AGENT_URL}/runs", json=request_data)
        response.raise_for_status()
        run_data = response.json()
        thread_id = run_data.get("thread_id")
        run_id = run_data["run_id"]
        print(f"Run ID: {run_id}, Thread ID: {thread_id}")

        # 2️⃣ Join Run Stream으로 실시간 스트리밍
        headers = {"Last-Event-ID": "-1"}  # 모든 이벤트부터 스트리밍
        async with client.stream('GET', f"{PUBV2AGENT_URL}/threads/{thread_id}/runs/{run_id}/stream", headers=headers) as stream_response:
            stream_response.raise_for_status()

            final_result = None
            async for line in stream_response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # "data: " 제거
                    if data == "[DONE]":
                        break
                    print(f"📊 raw 데이터 길이: {len(data)}자")

                    try:
                        import json
                        event = json.loads(data)

                        # 이벤트 타입에 따라 처리
                        event_type = event.get("event", "")
                        print(f"🔔 이벤트 타입: {event_type}")

                        if len(str(event)) > 1000:
                            print(f"   데이터 크기: {len(str(event))} characters")
                            print(f"   미리보기: {str(event)[:200]}...")
                            print(f"   키들: {list(event.keys())}")
                            if 'generated_response' in event:
                                print(f"     {event['generated_response']}")

                        else:
                            print(f"   데이터: {event}")

                    except json.JSONDecodeError as e:
                        print(f"❌ JSON 파싱 실패! 데이터 길이: {len(data)}자")
                        print(f"   에러: {e}")
                        print(f"   데이터 미리보기: {data[:200]}...")
                        print(f"   데이터 끝부분: ...{data[-200:]}")

                        # 중괄호 개수 체크
                        open_braces = data.count('{')
                        close_braces = data.count('}')
                        print(f"   중괄호 - 여는것: {open_braces}, 닫는것: {close_braces}")
                        continue


async def main():
    print("=== Join Run Stream 방식 ===")
    await run_pub_v2_agent_stream("미래에셋증권 자사주 소각 진짜야? 25년 2분기에 소각했는지 보고 그거때문에 9월 첫주에 둘째주보다 주가가 올랐는지 확인해줘")

if __name__ == "__main__":
    asyncio.run(main())
