from langchain.agents import create_agent
from llm import create_llm
from shared_tools import get_job_recommendations, get_resume_data
from system_prompts import SYSTEM_PROMPT_LOCAL_TOOLS
import gradio

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
    # print(response)
    print(response["messages"][-1].content)
    return response["messages"][-1].content

def deploy_agent():
    with gradio.Blocks(title="Jon Finder Agent") as iface:
        gradio.Markdown("## Job Finder Agent")
        gradio.Markdown("Finds job openings matched to your resume, using live listings and your indexed resume data.")

        job_title = gradio.Textbox(label="Job Title / Keywords", placeholder="python")
        salary = gradio.Textbox(label="Minimum Annual Salary (INR)", placeholder="4000000")

        find_jobs_btn = gradio.Button("Find Jobs", variant="primary")
        output = gradio.Textbox(label="Recomomended jobs", lines=20)

        find_jobs_btn.click(fn=run_agent, inputs=[job_title, salary], outputs=output)

    iface.launch()



if __name__ == '__main__':
    # response = run_agent("Python", 320000)
    deploy_agent()