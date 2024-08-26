import os
import openai
import argparse 
import yaml
from dotenv import load_dotenv, find_dotenv # pip install python-dotenvs
import logging
import sys,os,glob
from gen_tools.tools import bcolors, init_logger,str2bool,get_cmap # , warn_function
from langchain.prompts import ChatPromptTemplate
import getpass

# DEPRECATED: from langchain.chat_models import ChatOpenAI
# # DEPRECATED: from langchain_community.chat_models import ChatOpenAI

# pip install -U langchain-openai
from langchain_openai import ChatOpenAI


def get_completion(prompt, model, option, template = None):
    messages = [{"role": "user", "content": prompt}]
    

    if option=="openai":
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0, 
        )
        response = response.choices[0].message["content"]
    
    elif option=="langchain":
        if model.split("-")[0]=="gpt":
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

            # 1. BASIC INVOCATION 
            log.info(bcolors.WARNING + " ======================================== 1. BASIC INVOCATION ======================================== " + bcolors.WHITE)
            messages = [
                (
                    "system",
                    "You are a helpful assistant that translates English to French. Translate the user sentence.",
                ),
                ("human", "I love programming."),
            ]

            log.info(bcolors.OKGREEN + "(messages): " + bcolors.WHITE + str(messages))

            response = llm.invoke(messages)
            log.info(bcolors.OKGREEN + "(response) complete: " + bcolors.WHITE + str(response))
            log.info(bcolors.OKGREEN + "(response) content: " + bcolors.WHITE + str(response.content))


            # 2. CHAINING 
            log.info(bcolors.WARNING + " ======================================== 2. CHAINING (V1)======================================== " + bcolors.WHITE)
            messages = [
                            (
                                "system",
                                "You are a helpful assistant that translates {input_language} to {output_language}.",
                            ),
                            ("human", "{input}"),
                        ]

            prompt = ChatPromptTemplate.from_messages(messages)
            log.info(bcolors.OKGREEN + "prompt: " + bcolors.WHITE + str(prompt))

            chain = prompt | llm
            response = chain.invoke(
                {
                    "input_language": "English",
                    "output_language": "German",
                    "input": "I love programming.",
                })
            
            log.info(bcolors.OKGREEN + "(response) complete: " + bcolors.WHITE + str(response))
            log.info(bcolors.OKGREEN + "(response) content: " + bcolors.WHITE + str(response.content))


            # 3. CHAINING  - With Template from Template 
            log.info(bcolors.WARNING + " ======================================== 3. CHAINING (V2)======================================== " + bcolors.WHITE)


            template_string = """Translate the text \
            that is delimited by triple backticks \
            from {input_language} to {output_language} ```{input}```
            """

            prompt = ChatPromptTemplate.from_template(template_string)

            # Original Prompt -> prompt_template.messages[0].prompt
            # input variables -> prompt_template.messages[0].prompt.input_variables

            log.info(bcolors.OKGREEN + "prompt: " + bcolors.WHITE + str(prompt))

            chain = prompt | llm
            response = chain.invoke(
                {
                    "input_language": "English",
                    "output_language": "German",
                    "input": "I love programming.",
                })
            
            log.info(bcolors.OKGREEN + "(response) complete: " + bcolors.WHITE + str(response))
            log.info(bcolors.OKGREEN + "(response) content: " + bcolors.WHITE + str(response.content))



            # 4. REFORMATING PROMPT TEMPLATE    
            log.info(bcolors.WARNING + " ======================================== 3. CHAINING: REFORMATING PROMPT TEMPLATES ======================================== " + bcolors.WHITE)
            
            prompt_finetuned = prompt.format_messages(
                output_language="French",
                input_language = "English",
                input="Hola me llamo Johny")

            service_response = llm.invoke(prompt_finetuned)
            log.info(bcolors.OKGREEN + "(service_response) complete: " + bcolors.WHITE + str(service_response))
            log.info(bcolors.OKGREEN + "(service_response) content: " + bcolors.WHITE + str(service_response.content))



            breakpoint()

            


    return  response



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

    _ = load_dotenv(find_dotenv()) # read local .env file

    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

    openai.api_key = os.environ['OPENAI_API_KEY']
    log.info(bcolors.OKGREEN + "open_ai key: " + bcolors.WHITE + str(openai.api_key))

    customer_email = """
    Arrr, I be fuming that me blender lid \
    flew off and splattered me kitchen walls \
    with smoothie! And to make matters worse,\
    the warranty don't cover the cost of \
    cleaning up me kitchen. I need yer help \
    right now, matey!
    """

    style = """American English \
    in a calm and respectful tone
    """

    prompt = f"""Translate the text \
    that is delimited by triple backticks 
    into a style that is {style}.
    text: ```{customer_email}```
    """

    log.trace(bcolors.WARNING + "prompt: " + bcolors.WHITE + str(prompt))
    response = get_completion(prompt, llm_model, llm_option, template = None )
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
