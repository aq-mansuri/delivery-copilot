# scratch.py
import asyncio, json, os
from dotenv import load_dotenv
from app.integrations.atlassian.client import JiraClient

async def main():
    load_dotenv()
    c = JiraClient(
        base_url=os.environ["ATLASSIAN_BASE_URL"],
        email=os.environ["ATLASSIAN_EMAIL"],
        api_token=os.environ["ATLASSIAN_API_TOKEN"],
        target_release_field=os.environ["JIRA_TARGET_RELEASE_FIELD"],
    )
    raw = await c._get("/rest/api/3/issue/INS-2", {"expand": "changelog"})
    print(json.dumps(raw["fields"]["issuelinks"], indent=2))
    print(json.dumps(raw["changelog"]["histories"][:2], indent=2))

   
    await c.aclose()

asyncio.run(main())