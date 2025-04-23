import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from query_engine import create_query_engine

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def create_car_agent():
    # Set up the language model
    llm = ChatOpenAI(
        model="gpt-4",  # Using more capable model for agent
        temperature=0.2
    )
    
    # Create the query engine
    qa_chain = create_query_engine()
    
    # Define tools for the agent
    @tool
    def search_car_database(query: str) -> str:
        """Search the car database for information about specific cars or to compare cars based on their specifications."""
        result = qa_chain({"query": query})
        return result["result"]
    
    @tool
    def calculate_efficiency(weight: float, horsepower: float) -> str:
        """Calculate efficiency ratio (weight to horsepower) for a car."""
        ratio = weight / horsepower if horsepower > 0 else 0
        return f"The weight to horsepower ratio is {ratio:.2f} lbs/HP. Lower values indicate better performance."
    
    @tool
    def recommend_cars(criteria: str) -> str:
        """Recommend cars based on user criteria like 'high MPG', 'powerful engine', etc."""
        # This actually uses the same RAG system but with a recommendation framing
        query = f"Recommend cars that have {criteria}"
        result = qa_chain({"query": query})
        return result["result"]
    
    tools = [
        search_car_database,
        recommend_cars
    ]
    
    # Set up memory so agent remembers conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Initialize the agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )
    
    return agent

if __name__ == "__main__":
    agent = create_car_agent()
    
    # Example conversation
    while True:
        user_input = input("\nAsk about cars (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        
        response = agent.run(user_input)
        print(f"\nAgent: {response}") 