from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

model = ChatOllama(model="llama3:instruct")

prompt_template = """You are a E-shopping assistant chatbot to help costumers get the price of their desired products based on {input}. 
Respond should be based the following context, just respond we don't have the price of the product if it is not listed in the context.
<Context>
product: apple , price: $1.29
product: samsung s20 , price: $2.49
product: iphone 12 , price: $10.99
</Context>
ANSWER:
"""

prompt = ChatPromptTemplate.from_template(template=prompt_template)

chain = {"input": RunnablePassthrough()} | prompt | model
res = chain.invoke(" what is the price of xiaomi?")
print(res.content)