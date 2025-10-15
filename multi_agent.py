from dotenv import load_dotenv
import os

load_dotenv()

import httpx
import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

from langchain_core.prompts import PromptTemplate
from langchain_naver import ChatClovaX


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Multi-Agent API",
    description="PubAgent와 FinAgent를 통합한 멀티 에이전트 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 에이전트 서버 설정 (환경변수로 설정 가능)
FINAGENT_PORT = os.getenv("FINAGENT_PORT", "8000")
PUBAGENT_PORT = os.getenv("PUBAGENT_PORT", "6000")
PUBV2AGENT_PORT = os.getenv("PUBV2AGENT_PORT", "6200")
RUMAGENT_PORT = os.getenv("RUMAGENT_PORT", "9000")
RUMMULTIAGENT_PORT = os.getenv("RUMMULTIAGENT_PORT", "2025")
MULTIAGENT_PORT = int(os.getenv("MULTIAGENT_PORT", "10000"))

FINAGENT_URL = f"http://localhost:{FINAGENT_PORT}"
PUBAGENT_URL = f"http://localhost:{PUBAGENT_PORT}"
PUBV2AGENT_URL = f"http://localhost:{PUBV2AGENT_PORT}"
# RUMAGENT_URL = f"http://localhost:{RUMAGENT_PORT}"
RUMAGENT_URL = f"http://localhost:{RUMMULTIAGENT_PORT}"
# ClovaX LLM 설정
CLOVASTUDIO_API_KEY = os.getenv("CLOVASTUDIO_API_KEY", "")

llm = ChatClovaX(
    model="HCX-007",
    thinking={"effort": "none"},
    api_key=CLOVASTUDIO_API_KEY,
)

# 질문 유형 판단용 프롬프트
question_classifier_template = PromptTemplate(
    input_variables=["query"],
    template="""
당신은 한국 주식 전문가입니다.
사용자가 입력한 질문을 아래 4가지 유형 중 정확히 하나로 분류하세요.
반드시 JSON 형식으로만 출력해야 합니다.

1. 공시: 기업 공시, 실적, 임상시험, 배당, M&A 등 공식 공시 기반 정보
   - 예시: "하닉스 실적 언제 발표돼?", "카뱅 IPO 관련 공시 확인해줘"

2. 루머체크: 특정 기업 관련 소문, 루머, 뉴스 진위 여부 확인
   - 예시: "LG화학 배터리 관련 루머 진짜야?", "현대차 중국 진출설 확인해줘"

3. 주가검색: 시스템에서 실제로 지원되는 종목/시장 데이터 조회만 가능
   - 가능한 내용:
     - 특정 종목 시가/고가/저가/종가/거래량/등락률 조회
     - 시장 지수 조회 (KOSPI/KOSDAQ)
     - 시장 통계 조회
     - 가격/거래량/등락률 순위 조회
     - 기술적 지표 신호 조회 (RSI, 볼린저밴드, 이동평균선, 골든/데드크로스 등)
     - 복합조건 검색
   - 예시: "삼성전자 오늘 종가 얼마야?", "KOSPI 지수 오늘 얼마야?", "RSI 과매수 종목 알려줘"
   - ⚠️ PER, PBR, EPS 조회 및 투자 판단/추천/해석 관련 질문은 포함되지 않음

4. 기타: 투자 판단, 매수/매도 타이밍, 홀딩 여부, 물타기, 전략 관련 질문
   - 예시: "삼성전자 물타기 타이밍이야?", "카카오 하락장에서 홀딩해도 될까?"

질문: {query}

반드시 아래 형식으로 출력:
{{"type": "<공시/루머체크/주가검색/기타>"}}

- < > 안에는 하나의 유형만 들어가야 합니다.
- 절대 다른 형식이나 추가 텍스트를 넣지 마세요.
- 모호한 경우에도 반드시 위 네 가지 중 하나만 선택하세요.
- JSON 외 다른 텍스트를 출력하지 마세요.
"""
)

# 요청/응답 모델
class MultiAgentRequest(BaseModel):
    agent_type: str  # 'pub' or 'fin'
    query: str
    mode: Optional[str] = None  # PubAgent의 mode 분석용
    summary: Optional[str] = None  # PubAgent의 mode 분석용
    news_count: Optional[int] = 10  # FinAgent의 analyze용

class AutoSearchRequest(BaseModel):
    query: str

class MultiAgentResponse(BaseModel):
    agent_type: str
    result: Dict[Any, Any]
    success: bool
    error: Optional[str] = None
    question_type: Optional[str] = None  # 자동 분류된 질문 유형
    is_runid_only: bool = False

# 질문 유형별 에이전트 매핑
QUESTION_TYPE_TO_AGENT = {
    "공시": "pub",
    "루머체크": "rum",
    "주가검색": "fin",
    "기타": "pub"  # 투자 판단 관련은 기본적으로 fin으로
}

async def classify_question(query: str) -> str:
    """질문을 분류하여 적절한 에이전트 타입을 반환"""
    try:
        result = llm.invoke(question_classifier_template.format(query=query))

        # JSON 파싱
        response_text = result.content.strip()
        logger.info(f"질문 분류 결과: {response_text}")

        # JSON 추출 시도
        try:
            parsed = json.loads(response_text)
            question_type = parsed.get("type", "기타")
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 텍스트에서 직접 추출 시도
            if "공시" in response_text:
                question_type = "공시"
            elif "루머체크" in response_text:
                question_type = "루머체크"
            elif "주가검색" in response_text:
                question_type = "주가검색"
            else:
                question_type = "기타"

        agent_type = QUESTION_TYPE_TO_AGENT.get(question_type, "fin")
        logger.info(f"질문 '{query}' -> 유형: {question_type} -> 에이전트: {agent_type}")

        return question_type, agent_type

    except Exception as e:
        logger.error(f"질문 분류 중 오류: {str(e)}")
        # 오류 시 기본값으로 pub 사용
        return "기타", "pub"

@app.get("/")
async def root():
    """API 상태 확인"""
    return {
        "message": "Multi-Agent API Server",
        "status": "running",
        "version": "1.0.0",
        "available_agents": ["pub", "fin", "rum"]
    }

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    async with httpx.AsyncClient() as client:
        pub_status = "healthy"
        fin_status = "healthy"
        rum_status = "healthy"

        try:
            pub_response = await client.get(f"{PUBAGENT_URL}/health", timeout=5.0)
            if pub_response.status_code != 200:
                pub_status = "unhealthy"
        except Exception:
            pub_status = "unreachable"

        try:
            fin_response = await client.get(f"{FINAGENT_URL}/health", timeout=5.0)
            if fin_response.status_code != 200:
                fin_status = "unhealthy"
        except Exception:
            fin_status = "unreachable"

        try:
            rum_response = await client.get(f"{RUMAGENT_URL}/health", timeout=5.0)
            if rum_response.status_code != 200:
                rum_status = "unhealthy"
        except Exception:
            rum_status = "unreachable"

    return {
        "status": "healthy",
        "pubagent": pub_status,
        "finagent": fin_status,
        "rumagent": rum_status
    }

@app.post("/auto-search", response_model=MultiAgentResponse)
async def auto_search(request: AutoSearchRequest):
    """자동 에이전트 선택 - 질문을 분석해서 적절한 에이전트로 라우팅"""

    try:
        # 1. 질문 분류
        question_type, agent_type = await classify_question(request.query)

        # 2. 분류된 에이전트로 요청 생성
        agent_request = MultiAgentRequest(
            agent_type=agent_type,
            query=request.query
        )

        # 3. 해당 에이전트로 요청
        async with httpx.AsyncClient() as client:
            if agent_type == "pub":
                response = await _handle_pubagent_request(client, agent_request)
            elif agent_type == "rum":
                response = await _handle_rumagent_request(client, agent_request)
            else:
                response = await _handle_finagent_request(client, agent_request)

            # 4. 응답에 질문 유형 정보 추가
            response.question_type = question_type
            return response

    except Exception as e:
        logger.error(f"자동 검색 중 오류 발생: {str(e)}")
        return MultiAgentResponse(
            agent_type="unknown",
            result={},
            success=False,
            error=str(e),
            question_type="오류"
        )

@app.post("/search", response_model=MultiAgentResponse)
async def multi_agent_search(request: MultiAgentRequest):
    """멀티 에이전트 검색 - agent_type에 따라 분기처리 (수동 선택)"""

    if request.agent_type not in ["pub", "fin", "rum"]:
        raise HTTPException(
            status_code=400,
            detail="agent_type은 'pub', 'fin', 또는 'rum'이어야 합니다."
        )

    try:
        async with httpx.AsyncClient() as client:
            if request.agent_type == "pub":
                # PubAgent 요청
                return await _handle_pubagent_request(client, request)
            elif request.agent_type == "rum":
                # RumAgent 요청
                return await _handle_rumagent_request(client, request)
            else:
                # FinAgent 요청
                return await _handle_finagent_request(client, request)

    except Exception as e:
        logger.error(f"에이전트 요청 중 오류 발생: {str(e)}")
        return MultiAgentResponse(
            agent_type=request.agent_type,
            result={},
            success=False,
            error=str(e)
        )

# async def _handle_pubagent_request(client: httpx.AsyncClient, request: MultiAgentRequest) -> MultiAgentResponse:
#     """PubAgent 요청 처리"""
#     try:
#         # DART 공시 요약 요청
#         pub_request = {"query": request.query}
#         logger.info(f"PubAgent 요청: {pub_request}")
#         response = await client.post(
#             f"{PUBAGENT_URL}/summarize",
#             json=pub_request,
#             timeout=500.0
#         )
# 
#         if response.status_code == 200:
#             result_data = response.json()
#             return MultiAgentResponse(
#                 agent_type="pub",
#                 result=result_data,
#                 success=True
#             )
#         else:
#             raise HTTPException(
#                 status_code=response.status_code,
#                 detail=f"PubAgent 오류: {response.text}"
#             )
# 
#     except httpx.TimeoutException:
#         raise HTTPException(status_code=408, detail="PubAgent 응답 시간 초과")
#     except Exception as e:
#         logger.error(f"PubAgent 요청 오류: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"PubAgent 요청 실패: {str(e)}")

async def _handle_pubagent_request(client: httpx.AsyncClient, request: MultiAgentRequest) -> MultiAgentResponse:
    """PubAgent V2 요청 처리 (/pub_v2_wait 사용)"""
    try:
        request_data = {
            "assistant_id": "summarize",
            "input": {"query": request.query},
            "if_not_exists": "create"
        }

        logger.info(f"PubV2Agent wait 요청: {request_data}")
        response = await client.post(
            f"{PUBV2AGENT_URL}/runs/wait",
            json=request_data,
            timeout=500.0
        )

        if response.status_code == 200:
            result_data = response.json()
            summary = result_data.get("summary", "")
            need_finagent = result_data.get("need_finagent", False)
            final_output = result_data.get("final_output", {})

            # FinAgent 결과 포함 여부에 따라 최종 결과 구성
            final_generated = summary
            if need_finagent:
                fin_result = final_output.get("finagent_result", "")
                final_generated += f"\n\n# 📊 FlowAgent 결과:\n{fin_result}"

            return MultiAgentResponse(
                agent_type="pub",
                result={"summary": final_generated},
                success=True,
                is_runid_only=False
            )
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"PubV2Agent 오류: {response.text}"
            )

    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="PubV2Agent 응답 시간 초과")
    except Exception as e:
        logger.error(f"PubV2Agent 요청 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PubV2Agent 요청 실패: {str(e)}")



async def _handle_finagent_request(client: httpx.AsyncClient, request: MultiAgentRequest) -> MultiAgentResponse:
    """FinAgent 요청 처리"""
    try:
        # 주식 검색 요청
        fin_request = {"question": request.query}
        logger.info(f"FinAgent 요청: {fin_request}")
        response = await client.post(
            f"{FINAGENT_URL}/search",
            json=fin_request,
            timeout=500.0
        )
        logger.info(f"FinAgent 응답 상태: {response.status_code}")
        logger.info(f"FinAgent 응답 내용: {response.text}")
        if response.status_code == 200:
            result_data = response.json()
            return MultiAgentResponse(
                agent_type="fin",
                result=result_data,
                success=True
            )
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"FinAgent 오류: {response.text}"
            )

    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="FinAgent 응답 시간 초과")
    except Exception as e:
        logger.error(f"FinAgent 요청 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"FinAgent 요청 실패: {str(e)}")

async def _handle_rumagent_request(client: httpx.AsyncClient, request: MultiAgentRequest) -> MultiAgentResponse:
    """RumAgent 요청 처리 - 직접 HTTP 요청"""
    try:
        # 루머 검증 요청 - /runs/wait 엔드포인트 사용
        rum_request = {
            "assistant_id": "rum_multi_agent",
            "input": {
                "query": request.query
            },
            "if_not_exists": "create"
        }
        logger.info(f"RumAgent 요청: {rum_request}")
        response = await client.post(
            f"{RUMAGENT_URL}/runs/wait",
            json=rum_request,
            timeout=500.0
        )
        logger.info(f"RumAgent 응답 상태: {response.status_code}")
        logger.info(f"RumAgent 응답 내용: {response.text}")
        if response.status_code == 200:
            response_data = response.json()
            generated_response = response_data.get('generated_response', '')
            return MultiAgentResponse(
                agent_type="rum",
                result={"verification_result": generated_response},
                success=True
            )
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"RumAgent 오류: {response.text}"
            )

    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="RumAgent 응답 시간 초과")
    except Exception as e:
        logger.error(f"RumAgent 요청 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RumAgent 요청 실패: {str(e)}")

@app.post("/pub/analyze_mode")
async def pub_analyze_mode(query: str, mode: str, summary: str):
    """PubAgent의 모드별 분석 프록시"""
    try:
        async with httpx.AsyncClient() as client:
            logger.info(f"[pub_analyze_mode] 요청: query={query}, mode={mode}, summary={summary}")
            request_data = {
                "query": query,
                "mode": mode,
                "summary": summary
            }
            response = await client.post(
                f"{PUBAGENT_URL}/analyze_mode",
                json=request_data,
                timeout=500.0
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"PubAgent 분석 오류: {response.text}"
                )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PubAgent 분석 요청 실패: {str(e)}")


@app.post("/analyze_mode")
async def analyze_mode(request: dict):
    """모드별 추가 분석 - PubAgent로 프록시"""
    try:
        async with httpx.AsyncClient() as client:
            logger.info(f"[analyze_mode] 요청: {request}")
            response = await client.post(
                  f"{PUBAGENT_URL}/analyze_mode",
                  json=request,
                  timeout=500.0
            )

            if response.status_code == 200:
                  return response.json()
            else:
                  raise HTTPException(
                      status_code=response.status_code,
                      detail=f"PubAgent 분석 오류: {response.text}"
                  )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PubAgent 분석 요청 실패: {str(e)}")


@app.post("/fin/analyze")
async def fin_analyze(stock_name: str, news_count: int = 10):
    """FinAgent의 주식 분석 프록시"""
    try:
        async with httpx.AsyncClient() as client:
            request_data = {
                "stock_name": stock_name,
                "news_count": news_count
            }
            response = await client.post(
                f"{FINAGENT_URL}/analyze",
                json=request_data,
                timeout=500.0
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"FinAgent 분석 오류: {response.text}"
                )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FinAgent 분석 요청 실패: {str(e)}")

@app.post("/pub_v2")
async def pub_v2(request: AutoSearchRequest):
    """PubAgent V2 - run_id와 thread_id만 반환"""
    try:
        async with httpx.AsyncClient() as client:
            request_data = {
                "assistant_id": "summarize",
                "input": {"query": request.query},
                "if_not_exists": "create",
                "stream_resumable": True
            }
            logger.info(f"PubV2Agent 요청: {request_data}")
            response = await client.post(
                f"{PUBV2AGENT_URL}/runs",
                json=request_data,
                timeout=300.0
            )

            if response.status_code == 200:
                run_data = response.json()
                return {
                    "run_id": run_data["run_id"],
                    "thread_id": run_data.get("thread_id"),
                    "success": True
                }
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"PubV2Agent 오류: {response.text}"
                )

    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="PubV2Agent 응답 시간 초과")
    except Exception as e:
        logger.error(f"PubV2Agent 요청 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PubV2Agent 요청 실패: {str(e)}")

@app.post("/pub_v2_wait")
async def pub_v2_wait(request: AutoSearchRequest):
    """PubAgent V2 - wait 방식으로 결과까지 받아오기"""
    try:
        async with httpx.AsyncClient() as client:
            request_data = {
                "assistant_id": "summarize",
                "input": {"query": request.query},
                "if_not_exists": "create"
            }
            logger.info(f"PubV2Agent wait 요청: {request_data}")
            response = await client.post(
                f"{PUBV2AGENT_URL}/runs/wait",
                json=request_data,
                timeout=500.0
            )

            if response.status_code == 200:
                result_data = response.json()
                print(result_data.keys())
                summary = result_data.get('summary', '')
                need_finagent = result_data.get('need_finagent', False)
                final_output = result_data.get('final_output', '')
                final_generated = f"sumamry"
                if need_finagent:
                    final_generated += f"\n\n#\n{final_output.get('finagent_result', '')}"

                return {
                    "result": final_generated,
                    "success": True
                }
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"PubV2Agent wait 오류: {response.text}"
                )

    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="PubV2Agent wait 응답 시간 초과")
    except Exception as e:
        logger.error(f"PubV2Agent wait 요청 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PubV2Agent wait 요청 실패: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Multi-Agent API Server 시작")
    print(f"📡 PubAgent: {PUBAGENT_URL}")
    print(f"💰 FinAgent: {FINAGENT_URL}")
    print(f"🔍 RumAgent: {RUMAGENT_URL}")
    print(f"🔗 Multi-Agent: http://localhost:{MULTIAGENT_PORT}")
    print(f"📖 Swagger UI: http://localhost:{MULTIAGENT_PORT}/docs")
    print(f"📋 ReDoc: http://localhost:{MULTIAGENT_PORT}/redoc")

    uvicorn.run(app, host="0.0.0.0", port=MULTIAGENT_PORT)

"""
- uvicorn으로 실행:
nohup uvicorn multi_agent:app --host 0.0.0.0 --port 10000 --log-level debug > server.log 2>&1 &
nohup uvicorn multi_agent:app --host 0.0.0.0 --port 10000 --log-level debug > server_251014.log 2>&1 &

"""