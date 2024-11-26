
# Path Handling Libraries 
import argparse 
import yaml
import logging
import sys,os,glob
from tqdm import tqdm 
from datetime import date

# Environment Handling Libraries 
from dotenv import load_dotenv, find_dotenv # pip install python-dotenvs
import getpass
import numpy as np

# Import Gen Tools
from gen_tools.tools import bcolors, init_logger,str2bool,get_cmap # , warn_function

# LLM - Related Libraries
import openai
from openai import OpenAI

import langchain
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.chains import RetrievalQA

from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI


# Specific for Tutorial =========================================================================================
from langchain.agents import load_tools, initialize_agent, tool,\
 AgentExecutor, create_structured_chat_agent, create_react_agent
from langchain import hub
from langchain.agents import AgentType

from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool, PythonREPL


def get_completion(model, option, temperature = 0, execution_stages =[]):
    # messages = [{"role": "user", "content": prompt}]
        
    if option=="langchain":

        if model.split("-")[0]=="gpt":
            gpt_family  = True

            # To control the randomness and creativity of the generated
            # text by an LLM, use temperature = 0.0
            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
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
                temperature=temperature,
                huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
                add_to_git_credential=True,  # This saves the token to the Git credential helper
                **kwargs
            )
    else:
        raise(AssertionError("Choose another option (The only valid answer is 'langchain')"))

    tools = load_tools(["llm-math","wikipedia"], llm=llm)

    if "wikipedia" in execution_stages:


        # Agent: CHAT_ZERO_SHOT_REACT_DESCRIPTION : Works well with ChatModels 
            # LLM output -> Parsed 
            # LLM output -> Parsing Error -> Back to LLM to correct itself (handle_parsing_errors option to True )
        
        # [Notes] ¡Old!
        # agent = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True, verbose = True)

        # [Notes] New 
        # prompt = hub.pull("hwchase17/structured-chat-agent")
        # agent = create_structured_chat_agent(llm, tools, prompt)
        # agent_executor = AgentExecutor(agent=agent, tools=tools)
        # agent_executor.invoke({"input": question})

        # [Notes] New 

        """ 
        Source: https://github.com/langchain-ai/langchain/discussions/26751
        The verbose=True parameter causing the error with the new Pydantic version suggests 
        that the verbose logging might be interacting poorly with the callback handlers. 
        You can try removing or setting verbose=False to avoid this issue
        """

        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose = True)
        
        question = "If you have a rectangular garden with dimensions 12 meters by 8 meters, what is the total area of the garden in square meters?"
        result = agent_executor.invoke({"input": question})
        log.info(bcolors.OKGREEN + "(agent) question: " + bcolors.WHITE + str(question))
        log.info(bcolors.OKGREEN + "(agent) answer: " + bcolors.WHITE + str(result))
        
        question = "Andrew Ng is a British-American computer scientist and the Founder of Coursera and Deep Learning.ai and a Professor at Stanford University what books did he write?"
        result = agent_executor.invoke({"input": question})

        log.info(bcolors.OKGREEN + "(agent) question: " + bcolors.WHITE + str(question))
        log.info(bcolors.OKGREEN + "(agent) answer: " + bcolors.WHITE + str(result))
        
    if "python" in execution_stages:
        
        # [Notes] Another way of setting up to True 
        # langchain.debug=True

        agent = create_python_agent(
        llm,
        tool=PythonREPLTool(),
        verbose=True
         )
        

        employee_list = [["Smith", "John", 35], 
                        ["Doe", "Jane", 28],
                        ["Black", "Michael", 42],
                        ["Brown", "Emily", 31], 
                        ["White", "David", 39], 
                        ["Green", "Sarah", 45],
                        ["Jones", "Christopher", 37]
                        ]
        
        # [Run] This prompt is not working properly as it only perform one compound sorting using Last_name & Age
        # input=f""" Sort these employees by their age in ascending order and then by last name in descending order, and print the output: {employee_list}"""
        
        # [Notes] action_input -> Directly the code 

        input=f"""First sort these employees by their age in ascending order and print the result and then order them by last name in descending order, and print the output: {employee_list}"""
        agent.invoke({"input": input})

    if "custom_agent" in execution_stages:

        # CUSTOM AGENT

        @tool
        def time(text: str) -> str:
            """Returns todays date, use this for any \
            questions related to knowing todays date. \
            The input should always be an empty string, \
            and this function will always return todays \
            date - any date mathmatics should occur \
            outside this function."""
            return str(date.today())

        tools.append(time)

        from langchain.agents.react.output_parser import ReActOutputParser
        
        class CustomReActOutputParser(ReActOutputParser):
            def parse(self, text: str):
                if "Final Answer" in text and "Action" in text:
                    # Handle ambiguity (e.g., prioritize "Final Answer")
                    print(f'(before) final_answer: {final_answer}')
                    final_answer = text.split("Final Answer:")[1].strip().split("\n")[0]
                    print(f'(after) final_answer: {final_answer}')
                    return {"final_answer": final_answer}
                return super().parse(text)
    
        # agent = initialize_agent(
        #     tools,
        #     llm, 
        #     agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
        #     handle_parsing_errors=True, 
        #     verbose=True
        # )

        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt)

        custom_parser = CustomReActOutputParser()
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose = True, handle_parsing_errors = False, output_parser=custom_parser) #  max_iterations=2)

        # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose = True, handle_parsing_errors = False) #  max_iterations=2)
        question = "whats the date today?"
        result = agent_executor.invoke({"input": question})
        log.info(bcolors.OKGREEN + "(agent) question: " + bcolors.WHITE + str(question))
        log.info(bcolors.OKGREEN + "(agent) answer: " + bcolors.WHITE + str(result))


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
    execution_stages = ["custom_agent"]  # TODO: HARDCODED

    _ = load_dotenv(find_dotenv()) # read local .env file

    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

    if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your HUGGINFACEHUB API key: ")

    openai.api_key = os.environ['OPENAI_API_KEY']
    log.info(bcolors.OKGREEN + "open_ai key: " + bcolors.WHITE + str(openai.api_key))
    log.info(bcolors.OKGREEN + "HUGGINGFACEHUB_API_TOKEN: " + bcolors.WHITE + str(os.environ["HUGGINGFACEHUB_API_TOKEN"]))

    # [Notes] Math Tool:  is a chain itself -> Language Model + Calculator to solve math problems
    # [Notes] Wikipedia Tool:  API: Connects to Wikipedia database, run search queries against Wikipedia and retrieve results 
    get_completion(llm_model, llm_option, temperature = 0, execution_stages = execution_stages)

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
