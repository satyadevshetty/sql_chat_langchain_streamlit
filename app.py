from langchain.sql_database import SQLDatabase
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
import streamlit as st


st.title("Talk to your data")
api_key = st.text_input("ChatGPT api_key")
db_string = st.text_input("db_string (example: mysql+pymysql://<username>:<password>@<host>/<dbname>[?<options>])")

if api_key:
    db = SQLDatabase.from_uri(db_string)
    toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0, openai_api_key=api_key))
    agent_executor = create_sql_agent(
        llm=OpenAI(temperature=0, streaming=True, openai_api_key=api_key),
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
else:
    st.write("Please input openai_api_key")


if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor.run(prompt, callbacks=[st_callback])
        
        st.write(response)
