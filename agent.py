from langchain.agents import create_agent
from llm import create_llm
from shared_tools import get_job_recommendations, get_resume_data
from system_prompts import SYSTEM_PROMPT_LOCAL_TOOLS

tools = [get_job_recommendations, get_resume_data]

llm = create_llm()

job_search_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT_LOCAL_TOOLS
)

def run_agent(job_title: str, salary: str):
    user_prompt = (
        f"I'm looking for {job_title} roles with a minimum annual salary "
        f"of {salary}. Find matching live job openings, check my resume "
        f"against the skills these jobs require, and give me a prep plan "
        f"for any skills I'm missing."
    )

    response = job_search_agent.invoke(
        {"messages": [{"role": "user", "content": user_prompt}]}
    )
    print(response)
    return response["messages"][-1].content


    if __name__ == '__main__':
        response = run_agent("Python", 320000)