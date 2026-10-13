from langchain_community.tools import DuckDuckGoSearchRun
import requests
import os
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

@tool
def get_job_recommendations(what: str, salary_min: int) -> str:
    """
    Searches live Indian job listings (Adzuna) by keyword and minimum
    salary.

    Input Parameters:
    what (str) - One or more SINGLE KEYWORDS, space-separated...
    slary_min (int) - Minimum ANNUAL salary in Indian Rupees (INR),
    e.g. 400000 for 4 LPA.

    Returns:
    JSON with a "results" list. Each job object includes: title,
    company, company.display_name, description, salary_min, salary_max (all INR),
    location.display_name, and redirect_url

    """

    url = (
        f'https://api.adzuna.com/v1/api/jobs/in/search/1'
        f'?app_id={os.getenv("ADZUNA_APP_ID")}&app_key={os.getenv("ADZUNA_API_KEY")}'
        f'&what={what}&salary_min={salary_min}'
    )

    response = requests.get(url)
    return response.json()




if __name__ == '__main__':
    # web_search_tool = DuckDuckGoSearchRun()
    # result = web_search_tool.invoke("LA 2028 Olympics")
    # print(result)

    # result = get_job_recommendations("Data Analyst", 300000)
    result = get_job_recommendations.invoke({"what": "Data Analyst", "salary_min": 300000})
    print(result)