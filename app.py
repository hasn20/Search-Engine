import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
# StreamlitCallbackHandler displays the agent's thoughts and actions in real-time
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

## Code

# Initialize Arxiv tool - limits to top 1 result with 200 character preview
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# Initialize Wikipedia tool - limits to top 1 result with 200 character preview
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# Initialize DuckDuckGo search tool to perform web searches
search = DuckDuckGoSearchRun(name="Search")

# Create Streamlit app interface
st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

# Create sidebar for API key input
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Initialize chat history in session state if it doesn't exist yet
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display all previous messages in the chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Process new user input if provided and append it to msgs
if prompt := st.chat_input(placeholder="What is machine learning?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Only execute search if API key is provided
    if api_key:
        # Initialize the LLM with Groq
        llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
        
        # Set up tools for the agent to use (search, arxiv, wikipedia)
        tools = [search, arxiv, wiki]
        
        # Initialize agent with ZERO_SHOT_REACT_DESCRIPTION agent type
        # This agent makes decisions based on current input only without considering chat history
        # To use chat history, consider AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION instead
        search_agent = initialize_agent(
            tools, 
            llm, 
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
            handling_parsing_errors=True
        )

        # Display assistant's response with the agent's thought process
        with st.chat_message("assistant"):
            # StreamlitCallbackHandler shows the agent's internal decision-making process
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            
            # Run the agent with current messages and capture its response
            response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
            
            # Add assistant's response to chat history
            st.session_state.messages.append({'role': 'assistant', "content": response})
            
            # Display the final response
            st.write(response)
    else:
        # Notify user if API key is missing
        st.error("Please enter your Groq API Key in the sidebar to continue.")
