# ~~~!!! Library Setup !!!~~~

import os
import re
from datetime import date, datetime

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_pinecone import PineconeVectorStore

from langchain_huggingface import HuggingFaceEmbeddings

from configparser import ConfigParser



# ~~~!!! Config Setup !!!~~~

config = ConfigParser()
config.read("config.ini")

chatbot_logs = config["Data"]["chatbot_logs"]
pc_index_name = config["Pinecone"]["pc_index"]
k_sim_search_num = config["Pinecone"]["k_sim_search_num"]
model_name = config["Model"]["model_name"]


config_key = ConfigParser()
config_key.read("key_config.ini")

SLACK_APP_TOKEN = config_key["Slack"]["slack_app_api_key"]
SLACK_BOT_TOKEN = config_key["Slack"]["slack_bot_api_key"]
OPENAI_API_TOKEN = config_key["OpenAI"]["openai_api_key"]
PINECONE_API_TOKEN = config_key["Pinecone"]["pinecone_api_key"]



# ~~~!!! Environment Variable Setup !!!~~~

os.environ["OPENAI_API_KEY"] = OPENAI_API_TOKEN
os.environ["PINECONE_API_KEY"] = PINECONE_API_TOKEN




app = App(token=SLACK_BOT_TOKEN)
lc_embeddings = HuggingFaceEmbeddings(model_name=model_name)
vector_store = PineconeVectorStore(index_name=pc_index_name, embedding=lc_embeddings)

ERROR_OUTPUT = "Sorry, there was an issue! Contact the admins :)"





# ~~~!!! Chatbot Function Helpers !!!~~~

#This will send the user's query to the vector database, get the closest sentences, 
# then send to OpenAI to structure a response and return the output
def get_chatbot_msg(user_query):
    similarity_search_res = vector_store.similarity_search(user_query, k=int(k_sim_search_num))

    context = ""

    #Puts all doc into a single string
    for doc in similarity_search_res:
        context += doc.page_content
        context += " "
    

    template = """Assistant is a large language model trained by OpenAI.

    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

    Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

    Additionally, use the following CONTEXT to answer the user's QUESTION.
    CONTEXT:
    {pr_context}

    Keep your answer ground in the facts of the CONTEXT.
    If the CONTEXT doesn\'t contain the facts to answer the QUESTION return \"We don't have documentation on this currently, but, from OpenAI:\" and then answer their question as best as you can. Do not give false information.

    QUESTION:
    {human_input}

    Assistant:"""

    prompt = ChatPromptTemplate.from_template(template)

    chatgpt_chain = prompt | OpenAI(temperature=0)
    
    output = chatgpt_chain.invoke({"pr_context":context, "human_input":user_query})

    return output


#This function will write a log to "chatbot_logs", based on the month/year
def chatbot_logging(message, query, output):
    today = date.today()
    formatted_date = today.strftime("%Y-%m")

    with open(f"{chatbot_logs}{formatted_date}.log", "a") as fout:
        fout.write(f"------------------{datetime.now()}------------------\n")
        fout.write(f"{message}\n\n")
        fout.write(f"User Question: {query}\n\n")
        fout.write(f"{output}\n\n")
        fout.write("\n\n\n")



# ~~~!!! Slack Chatbot App Functions !!!~~~

#This is the code the bot will run when it is messaged in its private App channel
@app.message(".*")
def message_handler(message, say):
    if "text" in message:
        user_query = message["text"]
    else:
        chatbot_logging(message, "error", ERROR_OUTPUT)
        say(ERROR_OUTPUT)
        return
    
    output = get_chatbot_msg(user_query)

    chatbot_logging(message, user_query, output)

    #This is what will be returned to the user
    say(output)


#This is the code the bot will run when it is @'ed in a channel
@app.event("app_mention")
def handle_app_mention_events(body, say):
    if body != None and "event" in body:
        try:
            user_query = body["event"]["blocks"][0]["elements"][0]["elements"][1]["text"]
        except Exception as e:
            say(ERROR_OUTPUT)
            chatbot_logging(body, "error", ERROR_OUTPUT)
            return
    else:
        chatbot_logging(body, "error", ERROR_OUTPUT)
        say(ERROR_OUTPUT)
        return
    
    output = get_chatbot_msg(user_query)

    chatbot_logging(body, user_query, output)

    #This is what will be returned to the user0
    say(output)




if __name__ == "__main__":
    # ~~~!!! Socket Handler !!!~~~
    SocketModeHandler(app, SLACK_APP_TOKEN).start()



