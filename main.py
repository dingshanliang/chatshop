from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="llama3:instruct")

prompt_template = """You are a E-shopping assistant chatbot to help costumers get the price of their desired products. 
If user doesn't provide a specific model but only the brand name, you can recommend the phone listed in the context related with the brand with a price.
Respond should be based the following context, just respond we don't have the price of the product if it is not listed in the context.
Never recommend a product or brand is not listed in the context.
desiired_product: {product_name}
<Context>
product: apple , price: $1.29
product: samsung s20 , price: $2.49
product: iphone 12 , price: $10.99
</Context>
ANSWER:
"""

prompt = ChatPromptTemplate.from_template(template=prompt_template)

price_query_chain = {"product_name": RunnablePassthrough()} | prompt | model
# res = chain.invoke(" what is the price of xiaomi?")
# print(res.content)


prompt_extract_product = """You are a Product name or Brand name extractor based on user input. 
ALways just responde with the product name or brand name. No need to include any other information. No more than two words.
user_input: {input}
ANSWER:
"""

product_extract_chain = (
    {"input": RunnablePassthrough()} 
    | ChatPromptTemplate.from_template(template=prompt_extract_product) 
    | model
    | {"product_name": StrOutputParser()}
    )
# product_name = product_extract_chain.invoke("I want to buy a xaiomi mobile phone.").content

final_chain = product_extract_chain | price_query_chain

res = final_chain.invoke("I want to buy iphone.")

print(res.content)