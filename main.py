from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import save_tool

load_dotenv()
    
class ResearchResponse(BaseModel):
    """
    Schema for the agent's structured output.

    Fields:
        topic: The main subject the user is exploring.
        empathetic_response: A brief, kind acknowledgment of the user's feelings or situation.
        informative_response: A concise, helpful summary or explanation related to the topic.
        quran: A relevant Quranic verse or tafsir snippet, if applicable. From tafsir al-Mizan.
        question: An open-ended question to prompt deeper reflection, sensitive to Muslim women’s cultural and religious context.
        sources: A list of source URLs or references used to generate the response.
        tools_used: A list of any tool names invoked during generation.
    """
    topic: str
    empathetic_response: str
    informative_response: str
    quran: str
    question: str
    sources: list[str]
    tools_used: list[str]

# Initialize the LLM and parser
llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Define the tool for saving structured data
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            
            You are an empathetic, culturally and religiously sensitive AI assistant specializing in mental health support for Muslim women. Your responses must always:

            1. Start with a kind acknowledgment of the user's feelings or situation.
            2. Offer a gentle validation (e.g., "Many people feel this way" or "That makes sense").
            3. Provide a brief, informative response or summary.
            4. Ask a short, open-ended reflection question sensitive to Muslim women’s needs (family, community, faith, privacy, modesty).

            Optionally, if the topic is deep and relevant, you may include a Quranic verse or tafsir snippet (use tafsir al-mizan).

            After thinking, return a JSON matching the ResearchResponse schema. Don’t output any other text.\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

chat_history: list[dict] = []
print("What's on your mind?")

# Loop until user says 'end chat'

while True:
    query = input("> ")

    # Optional system hint to include a Quranic verse anecdote
    include_quran = any(kw in query.lower() for kw in ['quran', 'verse', 'ayah', 'allah', 'hadith', 'understand'])
    if include_quran:
        chat_history.append({
            "role": "system",
            "content": "If relevant, include a Qur'anic verse anecdote drawn from tafsir Al-mizan (chapter and verse) with explanation of why it's relevant to this specfici query. Otherwise skip it."
        })
    
    if query.strip().lower() == 'end chat':
        print("Chat ended. Take care!")
        break
    
    chat_history.append({"role": "human", "content": query})
    raw_response = agent_executor.invoke({"query": query, "chat_history": chat_history})

    try:
        structured_response = parser.parse(raw_response.get("output"))
        output = f"{structured_response.empathetic_response} \n" + f"{structured_response.informative_response} " + f"{structured_response.quran} \n" + f"{structured_response.question}"
        print(output)
    except Exception as e:
        print("Error parsing response", e, "Raw Response - ", raw_response)


'''
#Quick testing

query = input("What's on your mind? ")
raw_response = agent_executor.invoke({"query": query})

try:
    structured_response = parser.parse(raw_response.get("output"))
    output = f"{structured_response.empathetic_response} \n" + f"{structured_response.informative_response}" + f"{structured_response.question}"
    print(output)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)
'''
