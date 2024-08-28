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






def get_completion(prompt, model, option, template = None):
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

        # 4. REFORMATING PROMPT TEMPLATE    
        log.info(bcolors.WARNING + " ======================================== 3. CHAINING: REFORMATING PROMPT TEMPLATES ======================================== " + bcolors.WHITE)
        prompt = ChatPromptTemplate.from_template(review_template)
        prompt_finetuned = prompt.format_messages(text=customer_review)

        log.info(bcolors.OKGREEN + "prompt_finetuned: " + bcolors.WHITE + str(prompt_finetuned))
        service_response = llm.invoke(prompt_finetuned)
        log.info(bcolors.OKGREEN + "(service_response) complete: " + bcolors.WHITE + str(service_response))
        if gpt_family: log.info(bcolors.OKGREEN + "(service_response) content: " + bcolors.WHITE + str(service_response.content))



        # 5. REFORMATING PROMPT TEMPLATE: RESPONSE SCHEMAS 
        log.info(bcolors.WARNING + " ======================================== 3. CHAINING: RESPONSE SCHEMAS  ======================================== " + bcolors.WHITE)

        gift_schema = ResponseSchema(name="gift",
                                    description="Was the item purchased\
                                    as a gift for someone else? \
                                    Answer True if yes,\
                                    False if not or unknown.")
        delivery_days_schema = ResponseSchema(name="delivery_days",
                                            description="How many days\
                                            did it take for the product\
                                            to arrive? If this \
                                            information is not found,\
                                            output -1.")
        price_value_schema = ResponseSchema(name="price_value",
                                            description="Extract any\
                                            sentences about the value or \
                                            price, and output them as a \
                                            comma separated Python list.")

        response_schemas = [gift_schema, 
                            delivery_days_schema,
                            price_value_schema]

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        log.info(bcolors.OKGREEN + "Format instructions: " + bcolors.WHITE + str(format_instructions))


        prompt = ChatPromptTemplate.from_template(template=review_template_2)
        prompt_finetuned = prompt.format_messages(text=customer_review, 
                                        format_instructions=format_instructions)
                
        response = llm.invoke(prompt_finetuned)

        if gpt_family: 
            log.info(bcolors.OKGREEN + "(response) content: " + bcolors.WHITE + str(response.content))
            output_dict = output_parser.parse(response.content)
        else:
            output_dict = output_parser.parse(response)
            log.info(bcolors.OKGREEN + "(response) content: " + bcolors.WHITE + str(response))

        log.info(bcolors.OKGREEN + "output_dict: " + bcolors.WHITE + str(output_dict))

        
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

    _ = load_dotenv(find_dotenv()) # read local .env file

    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

    if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your HUGGINFACEHUB API key: ")

    openai.api_key = os.environ['OPENAI_API_KEY']
    log.info(bcolors.OKGREEN + "open_ai key: " + bcolors.WHITE + str(openai.api_key))
    log.info(bcolors.OKGREEN + "HUGGINGFACEHUB_API_TOKEN: " + bcolors.WHITE + str(os.environ["HUGGINGFACEHUB_API_TOKEN"]))


    response = get_completion(None, llm_model, llm_option, template = None )
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
