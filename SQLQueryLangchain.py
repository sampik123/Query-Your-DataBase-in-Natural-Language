import streamlit as st
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase
import os
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
import time
from langchain.agents.agent_toolkits.sql.prompt import (
    SQL_FUNCTIONS_SUFFIX,
    SQL_PREFIX,
    SQL_SUFFIX,
)
from langchain.callbacks.base import BaseCallbackManager
from langchain.agents.agent_toolkits.sql.base import create_sql_agent
from typing import Optional, List, Dict, Any, Sequence

# Set OpenAI and SerpApi API keys
os.environ['OPENAI_API_KEY'] = "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"
os.environ["SERPAPI_API_KEY"] = "d7a3c3a53753df43a73197406264924597a7413f36edc7d02ae0cf8ecb789854"

# Initialize OpenAI model and tools
llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Initialize the agent with AgentType.ZERO_SHOT_REACT_DESCRIPTION
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# connect to the database
db_user = "root"
db_password = "sampiksonu"
db_host = "localhost"
db_name = "classicmodels"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

# Create a ChatOpenAI model
llm = ChatOpenAI(model_name="gpt-3.5-turbo")


# Create an SQLDatabaseToolkit using the database
toolkit = SQLDatabaseToolkit(db=db, llm=llm)



# Initialize the agent executor


agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    callback_manager=None,
    prefix="""
    You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    If the question does not seem related to the database, just return "I don't know" as the answer.
    """,
    suffix=None,
    format_instructions="""
    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    """,
    input_variables=None,
    top_k=10,
    max_iterations=15,
    max_execution_time=None,
    early_stopping_method='force',
    agent_executor_kwargs=None,
    handle_parsing_errors=True
)



# Create the Streamlit interface
st.title("Database Chatbot: Query your database in natural language")

user_input = st.text_input("Ask a question:")
if st.button("Submit"):
    if user_input:
        response = agent_executor.run(user_input)
        st.write(response)


if __name__ == "__main__":
    # Sidebar content
    st.sidebar.write("About")
    st.sidebar.write("This is a chatbot powered by Langchain and Streamlit.")
    st.sidebar.write("You can use it to query a SQL database using natural language.")
    st.sidebar.markdown("[Streamlit](https://streamlit.io/)")
    st.sidebar.markdown("[LangChain](https://python.langchain.com/)")
    st.sidebar.markdown("[OpenAI LLM](https://platform.openai.com/docs/models)")
    st.sidebar.write("Made by [Sampik Kumar Gupta](https://www.linkedin.com/in/sampik-gupta-41544bb7/)")
