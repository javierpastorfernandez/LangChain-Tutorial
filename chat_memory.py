import os
import openai
import argparse 
import yaml
from dotenv import load_dotenv, find_dotenv # pip install python-dotenvs
import logging
import sys,os,glob
from gen_tools.tools import bcolors, init_logger,str2bool,get_cmap # , warn_function

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

import getpass
import logging

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationTokenBufferMemory
from langchain.memory import ConversationSummaryBufferMemory

# DEPRECATED: from langchain.chat_models import ChatOpenAI
# DEPRECATED: from langchain_community.chat_models import ChatOpenAI

# from langchain_huggingface import HuggingFaceEndpoint
# DEPRECATED: pip install -U langchain-huggingface


# pip install -U langchain-community
# DEPRECATED from langchain.llms import HuggingFaceHub


# pip install -U langchain-huggingface
from langchain_huggingface import HuggingFaceEndpoint
# from langchain_community.llms import HuggingFaceHub
# pip install -U langchain-openai
from langchain_openai import ChatOpenAI


customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}
"""


review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""






def get_completion(prompt, model, option, template = None, execution_stages = []):
    messages = [{"role": "user", "content": prompt}]
    
    if option=="openai":
        gpt_family  = True

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0, 
        )
        response = response.choices[0].message["content"]
    
    elif option=="langchain":

        if model.split("-")[0]=="gpt":
            gpt_family  = True

            # To control the randomness and creativity of the generated
            # text by an LLM, use temperature = 0.0
            llm = ChatOpenAI(
                model=model,
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
                # base_url="...",
                # organization="...",
                # other params...
            )
        
        else:
            gpt_family  = False
            kwargs = {"max_length":128}

            """
            The token used for authentication with Hugging Face has not been saved to the Git credentials helper
            This will store the token securely, making future logins easier.
            
            The Credential is being stored locally 
            """

            llm = HuggingFaceEndpoint(
                repo_id=model,
                temperature=0.5,
                huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
                add_to_git_credential=True,  # This saves the token to the Git credential helper
                **kwargs
            )

        # 1. CONVERSATION CHAIN / CONVERSATION MEMORY     
        log.info(bcolors.WARNING + " ======================================== 1. CONVERSATION CHAIN / CONVERSATION MEMORY  ======================================== " + bcolors.WHITE)

        if "ConversationBufferMemory" in execution_stages:
            memory = ConversationBufferMemory()
        
            conversation = ConversationChain(
                llm=llm, 
                memory = memory, # memory is holding conversation memory 
                verbose=True
            )
            log.info(bcolors.OKGREEN + "conversation: " + bcolors.WHITE + str(conversation))

            """
            CONVERSATION: 
                -> Takes a prompt, and embedds it in a larger prompt, so that the agent looks for context and truthfully says it does not have any idea if so 

            Prompt after formatting:
            The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
            Current conversation:
            Human: Hi, my name is Youssef
            AI:
            """

            conversation_input = "Hi, my name is Youssef"
            log.info(bcolors.OKGREEN + "conversation_input: " + bcolors.WHITE + str(conversation_input))
            response = conversation.predict(input=conversation_input)
            log.info(bcolors.OKGREEN + "response: " + bcolors.WHITE + str(response))

            conversation_input = "What is 1+1?"
            log.info(bcolors.OKGREEN + "conversation_input: " + bcolors.WHITE + str(conversation_input))
            response = conversation.predict(input=conversation_input)
            log.info(bcolors.OKGREEN + "response: " + bcolors.WHITE + str(response))

            conversation_input = "What is my name?"
            log.info(bcolors.OKGREEN + "conversation_input: " + bcolors.WHITE + str(conversation_input))
            response = conversation.predict(input=conversation_input)
            log.info(bcolors.OKGREEN + "response: " + bcolors.WHITE + str(response))

            # Memory is what comes after "Current conversation:"
            log.info(bcolors.OKGREEN + "(conversation) memory: " + bcolors.WHITE + str(memory.buffer))
            log.info(bcolors.OKGREEN + "(conversation) memory -> loadMemoryVariables: " + bcolors.WHITE + str(memory.load_memory_variables({})))

            # Explictly adding more context in the memory; Great
            memory.save_context({"input": "Hi"}, {"output": "What's up"})

            # This is working for GPT, not working for other models
            final_prompt = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
                    Current conversation:
                    Human: Hi, my name is Youssef
                    AI: Hello Youssef! It's nice to meet you. How can I assist you today?
                    Human: What is 1+1?
                    AI: 1+1 equals 2. Is there anything else you would like to know?
                    Human: What is my name?
                    AI:"""


            log.info(bcolors.OKGREEN + "final_prompt: " + bcolors.WHITE + str(final_prompt))
            response = llm.invoke(final_prompt)

            if gpt_family: 
                log.info(bcolors.OKGREEN + "(response) content: " + bcolors.WHITE + str(response.content))
            else:
                log.info(bcolors.OKGREEN + "(response) content: " + bcolors.WHITE + str(response))


            """
            As the conversation becomes long, the amount of memory needed becomes long, and the cost of sending a lot of tokens to the LLM, 
            which usually charges based on the number of tokens it needs to process, will also become more expensive.

            So LangChain provides several convenient kinds of memory to store and accumulate the conversation.
            So far, weve been looking at the ConversationBufferMemory. Let us look at a different type of memory.

            """


        if "ConversationBufferWindowMemory" in execution_stages:
            # 3. Conversation Buffer Window Memory
            log.info(bcolors.WARNING + " ======================================== CONVERSATION BUFFER WINDOW MEMORY  ======================================== " + bcolors.WHITE)

            memory = ConversationBufferWindowMemory(k=1) # Only one exchange human, AI 
            memory.load_memory_variables({}) 

            memory.save_context({"input": "Hi"},{"output": "What's up"})
            memory.save_context({"input": "Not much, just hanging"},{"output": "Cool"})

            # Memory is what comes after "Current conversation:"
            log.info(bcolors.OKGREEN + "(conversation) memory: " + bcolors.WHITE + str(memory.buffer))
            log.info(bcolors.OKGREEN + "(conversation) memory -> loadMemoryVariables: " + bcolors.WHITE + str(memory.load_memory_variables({})))



        if "ConversationTokenBufferMemory" in execution_stages:
            # 4. Conversation Token Buffer Memory
            # pip install tiktoken
            """
            LLM cost depends on tokens, not in HUMAN / AI interaction 
            """
            log.info(bcolors.WARNING + " ======================================== CONVERSATION TOKEN BUFFER MEMORY  ======================================== " + bcolors.WHITE)

            memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=20)
            memory.save_context({"input": "AI is what?!"},
                                {"output": "Amazing!"})
            memory.save_context({"input": "Backpropagation is what?"},
                                {"output": "Beautiful!"})
            memory.save_context({"input": "Chatbots are what?"}, 
                                {"output": "Charming!"})

            log.info(bcolors.OKGREEN + "(conversation) memory: " + bcolors.WHITE + str(memory.buffer))
            log.info(bcolors.OKGREEN + "(conversation) memory -> loadMemoryVariables: " + bcolors.WHITE + str(memory.load_memory_variables({})))




        if "ConversationSummaryBufferMemory" in execution_stages:
            # 5. Conversation Summary Memory
            log.info(bcolors.WARNING + " ======================================== Conversation Summary Memory  ======================================== " + bcolors.WHITE)

            """let us use an LLM to write a summary of the conversation and let that be the memory.
            """

            # create a long string
            schedule = "There is a meeting at 8am with your product team. \
            You will need your powerpoint presentation prepared. \
            9am-12pm have time to work on your LangChain \
            project which will go quickly because Langchain is such a powerful tool. \
            At Noon, lunch at the italian resturant with a customer who is driving \
            from over an hour away to meet you to understand the latest in AI. \
            Be sure to bring your laptop to show the latest LLM demo."

            """
            Use an LLM to write a summary of the conversation and let that be the memory
                -> if the number of tokens does not overload -> The conversation is intact 
                -> if the number of tokens DOES overload -> We use an LLM to summary the conversation 
            """

            memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=400)
            memory.save_context({"input": "Hello"}, {"output": "What's up"})
            memory.save_context({"input": "Not much, just hanging"},
                                {"output": "Cool"})
            memory.save_context({"input": "What is on the schedule today?"}, 
                                {"output": f"{schedule}"})

            log.info(bcolors.OKGREEN + "(conversation) memory: " + bcolors.WHITE + str(memory.buffer))
            log.info(bcolors.OKGREEN + "(conversation) memory -> loadMemoryVariables: " + bcolors.WHITE + str(memory.load_memory_variables({})))

            memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
            memory.save_context({"input": "Hello"}, {"output": "What's up"})
            memory.save_context({"input": "Not much, just hanging"},
                                {"output": "Cool"})
            memory.save_context({"input": "What is on the schedule today?"}, 
                                {"output": f"{schedule}"})

            log.info(bcolors.OKGREEN + "(conversation) memory: " + bcolors.WHITE + str(memory.buffer))
            log.info(bcolors.OKGREEN + "(conversation) memory -> loadMemoryVariables: " + bcolors.WHITE + str(memory.load_memory_variables({})))
            

            conversation = ConversationChain(
                llm=llm, 
                memory = memory,
                verbose=True
            )

            response = conversation.predict(input="What would be a good demo to show?")

            if gpt_family: 
                log.info(bcolors.OKGREEN + "(response) content: " + bcolors.WHITE + str(response.content))
            else:
                log.info(bcolors.OKGREEN + "(response) content: " + bcolors.WHITE + str(response))






        if "VectorDataMememory" in execution_stages:
            # 5. Conversation Summary Memory
            log.info(bcolors.WARNING + " ======================================== VectorDataMememory  ======================================== " + bcolors.WHITE)

            """
            VectorDataMemory:
                - > Works with embeddings (words / text )
                - > LangChain can retrieve the most relevant part of the text 
                - > Entire conversation stored in other databases such as SQL database or key/value database 
            """
        

    return response



def main(args):
    log = logging.getLogger('logger')

    plot_verbosity=args.plot_verbosity
    verbosity=args.verbosity
    config_filepath=args.config_filepath

    log.info(bcolors.OKGREEN + "plot_verbosity: " + bcolors.WHITE + str(plot_verbosity))
    log.info(bcolors.OKGREEN + "verbosity: " + bcolors.WHITE + str(verbosity))
    log.info(bcolors.OKGREEN + "config_filepath: " + bcolors.WHITE + str(config_filepath))

    with open(config_filepath) as f:
        config = yaml.safe_load(f)


    llm_option = config["llm"]["option"]
    log.info(bcolors.OKGREEN + "llm_option: " + bcolors.WHITE + str(llm_option))

    llm_model = config["llm"]["model"]
    log.info(bcolors.OKGREEN + "llm_model: " + bcolors.WHITE + str(llm_model))

    execution_stages = config["llm"]["stages"]
    log.info(bcolors.OKGREEN + "execution_stages: " + bcolors.WHITE + str(execution_stages))

    _ = load_dotenv(find_dotenv()) # read local .env file

    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

    if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your HUGGINFACEHUB API key: ")

    openai.api_key = os.environ['OPENAI_API_KEY']
    log.info(bcolors.OKGREEN + "open_ai key: " + bcolors.WHITE + str(openai.api_key))
    log.info(bcolors.OKGREEN + "HUGGINGFACEHUB_API_TOKEN: " + bcolors.WHITE + str(os.environ["HUGGINGFACEHUB_API_TOKEN"]))


    response = get_completion(None, llm_model, llm_option, template = None, execution_stages = execution_stages)
    log.trace(bcolors.WARNING + "response: " + bcolors.WHITE + str(response))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_verbosity',  type=int,    default=0, help='Plot results in console in console')
    parser.add_argument('--verbosity',     type=int,    default=1, help='Display messages in console')
    parser.add_argument('--config_filepath', type=str,default="data/config.yaml", help='Filepath of config file')
    args = parser.parse_args()
    
    # Initialize logger 
    _=init_logger("logger",args.verbosity)
    log = logging.getLogger('logger')
    log.trace("Checking new custom level")


    main(args)
