from vectara_agentic.agent import Agent
from vectara_agentic.agent_endpoint import start_app

from dotenv import load_dotenv
load_dotenv(override=True)

#customer_id = '1366999410'
corpus_key = 'vectara-docs_1'
api_key = 'zqt_UXrBcnI2UXINZkrv4g1tQPhzj02vfdtqYJIDiA'

assistant = Agent.from_corpus(
    tool_name = 'query_vectara_website',
    vectara_corpus_id = corpus_key,
    vectara_api_key = api_key,
    data_description = 'Data from vectara.com website',
    assistant_specialty = 'vectara'
)

start_app(assistant, host="0.0.0.0", port=8000)
