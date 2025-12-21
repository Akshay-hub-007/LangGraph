from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
load_dotenv()
# ---------- 1. Set your API Key ----------

# ---------- 2. Create the LLM ----------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash"
)

# ---------- 3. Create the Prompt with a MessagesPlaceholder ----------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history"),  # placeholder for past messages
    ("human", "{input}")                   # latest user message
])

# ---------- 4. Create Memory to store conversation ----------
memory = ConversationBufferMemory(
    memory_key="chat_history",  # matches placeholder name
    return_messages=True
)

# ---------- 5. Create the Chain ----------
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

# ---------- 6. Run the Conversation ----------
print("First call:")
print(chain.run(input="Hi, what's 2 + 2?"))

print("\nSecond call (with memory):")
print(chain.run(input="Multiply that by 5."))

print("\nThird call (with memory):")
print(chain.run(input="What is the square root of that?"))
