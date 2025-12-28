from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from src.tools import get_tools
from src.config import GOOGLE_API_KEY, MODEL_NAME

def initialize_agent():
    # 1. Google Gemini Modelini Tanımla
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )

    # 2. Araçları Al
    tools = get_tools()

    # 3. Hazır Prompt'u Çek (ReAct yapısı için)
    prompt = hub.pull("hwchase17/react")

    # 4. Ajanı Oluştur
    agent = create_react_agent(llm, tools, prompt)

    # 5. Çalıştırıcıyı (Executor) Hazırla
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, # Düşünme sürecini görmek için açık
        handle_parsing_errors=True
    )
    
    return agent_executor