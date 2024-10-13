import os
import openai
import argparse 
import yaml
from dotenv import load_dotenv, find_dotenv # pip install python-dotenvs
import logging
import sys,os,glob
from gen_tools.tools import bcolors, init_logger,str2bool,get_cmap # , warn_function

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

import getpass
import logging

from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
import pandas as pd

from langchain.chains import LLMChain
from langchain.chains import SequentialChain, SimpleSequentialChain

from langchain_core.runnables import RunnableParallel
from langchain_core.runnables.passthrough import RunnablePick
from langchain_core.output_parsers import StrOutputParser


from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
# from langchain_core.runnables.base import RunnableLambda
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from operator import itemgetter
from typing import Literal
from typing_extensions import TypedDict


def get_completion(prompt, model, option, template = None, execution_stages = []):
    messages = [{"role": "user", "content": prompt}]
    temperature = 0
    
    if option=="openai":
        gpt_family  = True

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature, 
        )
        response = response.choices[0].message["content"]
    
    elif option=="langchain":

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


        data_path =  './data/product_reviews.csv'
        df = pd.read_csv(data_path)
        log.info(bcolors.OKGREEN + "(df) head:\n" + bcolors.WHITE + str(df.head()))


        prompt = ChatPromptTemplate.from_template(
            "What is the best name to describe \
            a company that makes {product}?"
        )

        if "previous_knowledge" in execution_stages:

            # 0.  PREVIOUS CHAINING KNOWLEDGE 
            log.info(bcolors.WARNING + " ======================================== PREVIOUS CHAINING KNOWLEDGE======================================= " + bcolors.WHITE)


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

        if "llm_chain" in execution_stages:
            # 1. LLM CHAIN     
            log.info(bcolors.WARNING + " ======================================== 1. LLM CHAIN ================================================== " + bcolors.WHITE)

            prompt = ChatPromptTemplate.from_template(
                "What is the best name to describe \
                a company that makes {product}?"
            )
            chain = LLMChain(llm=llm, prompt=prompt) #deprecated
            product = "Deep Learning GPUs"
            response = chain.run(product)
            log.info(bcolors.OKGREEN + "(response) content: " + bcolors.WHITE + str(response))

            # Use instead 
            chain = prompt | llm
            response = chain.invoke(
                    {
                        "product": f"{product}"
                    })
            
            if gpt_family: 
                log.info(bcolors.OKGREEN + "(response) content: " + bcolors.WHITE + str(response.content))
            else:
                log.info(bcolors.OKGREEN + "(response) content: " + bcolors.WHITE + str(response))

        if "double_chain" in execution_stages:
            product = "Deep Learning GPUs"

            # 1. DOUBLE LLM CHAIN     
            log.info(bcolors.WARNING + " ======================================== 2. DOUBLE LLM CHAIN  ================================================== " + bcolors.WHITE)

            # prompt template 1
            first_prompt = ChatPromptTemplate.from_template(
                "What is the best name to describe \
                a company that makes {product}?"
            )

            # prompt template 2
            second_prompt = ChatPromptTemplate.from_template(
                "Write a 30 words description for the following \
                company:{company_name}"
            )


            # Interestingly, the company name changed when chaining the prompt two times 
            # Use instead 
            chain = first_prompt|second_prompt|llm
            response = chain.invoke(
                    {
                        "product": f"{product}"
                    })
            
            if gpt_family: 
                log.info(bcolors.OKGREEN + "(response) content: " + bcolors.WHITE + str(response.content))
            else:
                log.info(bcolors.OKGREEN + "(response) content: " + bcolors.WHITE + str(response))



            # Chain 1
            chain_one = LLMChain(llm=llm, prompt=first_prompt)

            # chain 2
            chain_two = LLMChain(llm=llm, prompt=second_prompt)

            overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
                                                        verbose=True)
            
            response = overall_simple_chain.run(product)
            log.info(bcolors.OKGREEN + "(response) content: " + bcolors.WHITE + str(response))

        if "complex_chain" in execution_stages:

            # 3. COMPLEX SEQUENTIAL CHAIN 
            log.info(bcolors.WARNING + " ======================================== COMPLEX SEQUENTIAL CHAIN  ================================================== " + bcolors.WHITE)
            
            # DATUM -------------------------------------------------
            review = df.Review[5]
            log.info(bcolors.OKGREEN + "review: " + bcolors.WHITE + str(review))
            review = "La estufa funcionó bien, pero pesa mucho para largas caminatas"

            # prompt template 1: translate to english
            first_prompt = ChatPromptTemplate.from_template(
                "Translate the following review to english:"
                "\n\n{Review}" # -> English_Review
            )

            second_prompt = ChatPromptTemplate.from_template(
                "Can you summarize the following review in 1 sentence:"
                "\n\n{English_Review}" # -> summary
            )

            # prompt template 3: translate to english
            third_prompt = ChatPromptTemplate.from_template(
                "What language is the following review:\n\n{Review}"
            ) # -> language 

            # prompt template 4: follow up message
            fourth_prompt = ChatPromptTemplate.from_template(
                "Write a follow up response to the following "
                "summary in the specified language:"
                "\n\nSummary: {summary}\n\nLanguage: {language}"
            ) # -> followup_message

            # METHOD 1 --------------------------
            # chain 1: input= Review and output= English_Review
            chain_one = LLMChain(llm=llm, prompt=first_prompt, 
                                output_key="English_Review"
                                )
            

            # chain 2: input= English_Review and output= summary
            chain_two = LLMChain(llm=llm, prompt=second_prompt, 
                                output_key="summary"
                                )
        
            # chain 3: input= Review and output= language
            chain_three = LLMChain(llm=llm, prompt=third_prompt,
                                output_key="language"
                                )

            # chain 4: input= summary, language and output= followup_message
            chain_four = LLMChain(llm=llm, prompt=fourth_prompt,
                                output_key="followup_message"
                                ) # in the original language 

            # overall_chain: input= Review 
            # and output= English_Review,summary, followup_message
            overall_chain = SequentialChain(
                chains=[chain_one, chain_two, chain_three, chain_four],
                input_variables=["Review"],
                output_variables=["English_Review", "summary","followup_message"],
                verbose=True
            )

            # Log the response
            response = overall_chain(review)
            log.info(bcolors.OKGREEN + "response: " + bcolors.WHITE + str(response))
  

            # METHOD 1B --------------------------
            code_prompt = ChatPromptTemplate.from_template("Write a very short {language} function that will {task}")
            chain_one = first_prompt | llm | {"English_Review": StrOutputParser()}
            chain_three = third_prompt | llm | {"language": StrOutputParser()}
            chain_two = second_prompt | llm | {"summary": StrOutputParser()}
            chain_four = fourth_prompt | llm | {"followup_message": StrOutputParser()}

            # PARALELL
                # chain1 (Review) -> (English_Review)
                # chain3 (Review) -> (Language)
            # chain2 (English_Review) -> (summary)
            # chain4 (language, summary) -> (followup)

            chain = RunnableParallel({
                "Review": RunnablePick("Review"),
                "English_Review": chain_one | RunnablePick("English_Review"),
                "language": chain_three | RunnablePick("language"),
            })|RunnableParallel({
                "language": RunnablePick("language"),
                "summary": chain_two | RunnablePick("summary"),
            })|RunnableParallel({
                "language": RunnablePick("language"),
                "summary": RunnablePick("summary"),
                "English_Review": RunnablePick("English_Review"),
                "followup_message": chain_four | RunnablePick("followup_message"),
            })
            response = chain.invoke({
                "Review": f"{review}"
            })

            # Log the response
            log.info(bcolors.OKGREEN + "response: " + bcolors.WHITE + str(response))

            breakpoint()

            # METHOD2
            code_prompt = ChatPromptTemplate.from_template("Write a very short {language} function that will {task}")
            code_chain = code_prompt | llm | {"code": StrOutputParser()}
            # code_chain = code_prompt | llm | {"code": lambda x: x}  # Direct output, no parsing

            test_prompt = ChatPromptTemplate.from_template("Write a test for the following {language} code:\n{code}")
            test_chain = test_prompt | llm | {"test": StrOutputParser()}

            # This does not work cause there is no correct post-processing
                # Example: code': AIMessage(content='def reverse_string(s):\n    return s[::-1]', addi
            # test_chain = test_prompt | llm | {"test": lambda x: x}  # Direct output, no parsing

            chain = RunnableParallel({
                "language": RunnablePick("language"),
                #Launches the chain, and then picks code result 
                "code": code_chain | RunnablePick("code"),
            }) | RunnableParallel({
                "language": RunnablePick("language"),
                "code": RunnablePick("code"),
                "test": test_chain | RunnablePick("test"),
            })

            response = chain.invoke({
                "language": "python",
                "task": "reverse a string",
            })

            log.info(bcolors.OKGREEN + "response: " + bcolors.WHITE + str(response))






            # Now Pass Chain Over Pandas columns 
            breakpoint()
            # IMPORTANT:
                # If review is already in English, there is actually some problems 
                # If not, it works well 



    


            # Interestingly, the company name changed when chaining the prompt two times 
            # Use instead 
            chain = first_prompt|second_prompt|llm
            response = chain.invoke(
                    {
                        "product": f"{product}"
                    })
            
            if gpt_family: 
                log.info(bcolors.OKGREEN + "(response) content: " + bcolors.WHITE + str(response.content))
            else:
                log.info(bcolors.OKGREEN + "(response) content: " + bcolors.WHITE + str(response))



            # Chain 1
            chain_one = LLMChain(llm=llm, prompt=first_prompt)

            # chain 2
            chain_two = LLMChain(llm=llm, prompt=second_prompt)

            overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
                                                        verbose=True)
            
            response = overall_simple_chain.run(product)
            log.info(bcolors.OKGREEN + "(response) content: " + bcolors.WHITE + str(response))



        if "router_chain_example" in execution_stages:


            prompt_1 = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are an expert on animals."),
                    ("human", "{query}"),
                ]
            )
            prompt_2 = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are an expert on vegetables."),
                    ("human", "{query}"),
                ]
            )

            log.info(bcolors.OKGREEN + "prompt_1: " + bcolors.WHITE + str(prompt_1))
            log.info(bcolors.OKGREEN + "prompt_2: " + bcolors.WHITE + str(prompt_2))

            chain_1 = prompt_1 | llm | StrOutputParser()
            chain_2 = prompt_2 | llm | StrOutputParser()

            route_system = "Route the user's query to either the animal or vegetable expert."
            route_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", route_system),
                    ("human", "{query}"),
                ]
            )


            class RouteQuery(TypedDict):
                """Route query to destination."""
                destination: Literal["animal", "vegetable"]


            route_chain = (
                route_prompt
                | llm.with_structured_output(RouteQuery)
                | itemgetter("destination")
            )

            chain = {
                "destination": route_chain,  # "animal" or "vegetable"
                "query": lambda x: x["query"],  # pass through input query
            } | RunnableLambda(
                # if animal, chain_1. otherwise, chain_2.
                lambda x: chain_1 if x["destination"] == "animal" else chain_2,
            )

            response = chain.invoke({"query": "what color are carrots"})
            log.info(bcolors.OKGREEN + "response: " + bcolors.WHITE + str(response))





        if "router_chain" in execution_stages:
            """
            Summary: The input can be driven to different LLM Routes
            Each Route is actually a prompt that modulates the LLM Route to what we want

                - You can route between prompt templates or chains 
                - You need to parse the output of the Router chain -> some will be re-directed to choose the path / the input of the chain itself
                - We will need also a default chain: A chain when no specialized GPT can be used for this purpose! 


            A good way to imagine this is if you have multiple subchains, each of which is specialized for a particular type of input
            """

            physics_template = """You are a very smart physics professor. \
            You are great at answering questions about physics in a concise\
            and easy to understand manner. \
            When you don't know the answer to a question you admit\
            that you don't know.

            Here is a question:
            {input}"""


            math_template = """You are a very good mathematician. \
            You are great at answering math questions. \
            You are so good because you are able to break down \
            hard problems into their component parts, 
            answer the component parts, and then put them together\
            to answer the broader question.

            Here is a question:
            {input}"""


            history_template = """You are a very good historian. \
            You have an excellent knowledge of and understanding of people,\
            events and contexts from a range of historical periods. \
            You have the ability to think, reflect, debate, discuss and \
            evaluate the past. You have a respect for historical evidence\
            and the ability to make use of it to support your explanations \
            and judgements.

            Here is a question:
            {input}"""


            computerscience_template = """ You are a successful computer scientist.\
            You have a passion for creativity, collaboration,\
            forward-thinking, confidence, strong problem-solving capabilities,\
            understanding of theories and algorithms, and excellent communication \
            skills. You are great at answering coding questions. \
            You are so good because you know how to solve a problem by \
            describing the solution in imperative steps \
            that a machine can easily interpret and you know how to \
            choose a solution that has a good balance between \
            time complexity and space complexity. 

            Here is a question:
            {input}"""


            prompt_infos = [
                {
                    "name": "physics", 
                    "description": "Good for answering questions about physics", 
                    "prompt_template": physics_template
                },
                {
                    "name": "math", 
                    "description": "Good for answering math questions", 
                    "prompt_template": math_template
                },
                {
                    "name": "History", 
                    "description": "Good for answering history questions", 
                    "prompt_template": history_template
                },
                {
                    "name": "computer science", 
                    "description": "Good for answering computer science questions", 
                    "prompt_template": computerscience_template
                }
            ]


            option = 2 
            destination_chains = {}
            for p_info in prompt_infos:
                name = p_info["name"]
                prompt_template = p_info["prompt_template"]
                prompt = ChatPromptTemplate.from_template(template=prompt_template)

                # Use instead 
                if option==1:
                    chain = LLMChain(llm=llm, prompt=prompt)
                elif option==2:
                    chain = prompt | llm

                destination_chains[name] = chain  # this is what is useful
                
            destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
            destinations_str = "\n".join(destinations)
            log.info(bcolors.OKGREEN + "(destinations_str): " + bcolors.WHITE + str(destinations_str))

            default_prompt = ChatPromptTemplate.from_template("{input}")

            if option==1:
                default_chain = LLMChain(llm=llm, prompt=default_prompt)
            
            if option==2:
                default_chain = default_prompt | llm  # Updated to use pipe operator


            MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
            language model select the model prompt best suited for the input. \
            You will be given the names of the available prompts and a \
            description of what the prompt is best suited for. \
            You may also revise the original input if you think that revising\
            it will ultimately lead to a better response from the language model.

            << FORMATTING >>
            Return a markdown code snippet with a JSON object formatted to look like:
            ```json
            {{{{
                "destination": string \ name of the prompt to use or "DEFAULT"
                "next_inputs": string \ a potentially modified version of the original input
            }}}}
            ```

            REMEMBER: "destination" MUST be one of the candidate prompt \
            names specified below OR it can be "DEFAULT" if the input is not\
            well suited for any of the candidate prompts.
            REMEMBER: "next_inputs" can just be the original input \
            if you don't think any modifications are needed.

            << CANDIDATE PROMPTS >>
            {destinations}

            << INPUT >>
            {{input}}

            << OUTPUT (remember to include the ```json)>>"""
            log.info(bcolors.OKGREEN + "(MULTI_PROMPT_ROUTER_TEMPLATE): " + bcolors.WHITE + str(MULTI_PROMPT_ROUTER_TEMPLATE))

            router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
                destinations=destinations_str
            )

            log.info(bcolors.OKGREEN + "(router_template): " + bcolors.WHITE + str(router_template))

            router_prompt = PromptTemplate(
                template=router_template,
                input_variables=["input"],
                output_parser=RouterOutputParser(),
            )


            if option==1:
                router_chain = LLMRouterChain.from_llm(llm, router_prompt)
            

            elif option==2:

                class RouteQuery(TypedDict):
                    """Route query to destination."""
                    destination: Literal["physics", "math", "History","computer science"]

                """
                route_chain = (
                    router_prompt
                    | llm.with_structured_output(RouteQuery)
                    | itemgetter("destination")
                )
                """

                route_chain = (
                    router_prompt
                    |llm 
                    |StrOutputParser()
                )

                def routing_logic(destination, next_inputs):
                    log = logging.getLogger('logger')
                    log.info(bcolors.OKGREEN + "destination: " + bcolors.WHITE + str(destination))
                    log.info(bcolors.OKGREEN + "next_inputs: " + bcolors.WHITE + str(next_inputs))

                    # Select the template based on the destination
                    if destination == "physics":
                        return destination_chains[destination]
                    elif destination == "math":
                        return destination_chains[destination]
                    elif destination == "History":
                        return destination_chains[destination]
                    elif destination == "computer science":
                        return destination_chains[destination]
                    else:
                        return default_chain    # Return default template for handling by the default chain


                """
                chain = {
                    "destination": route_chain,  # 
                    "next_inputs": lambda x: x["input"],  # pass input to the Route Chain -> should match to the RouterChain Input variables
                } | RunnableLambda(routing_lambda) |llm
                
                """

                breakpoint()
                # Define a RunnableLambda chain for routing
                routing_chain = RunnableLambda(
                    func=lambda x: routing_logic(x['destination'], x['next_inputs']), 
                )

                combined_chain = route_chain | routing_chain | llm

            
            if option==1:
                chain = MultiPromptChain(router_chain=router_chain, 
                                        destination_chains=destination_chains, 
                                        default_chain=default_chain, verbose=True)
                response = chain.run("What is Generative AI?")

            elif option==2:
                breakpoint()
                response = route_chain.invoke({"input": "What is Generative AI?"})
                log.info(bcolors.OKGREEN + "(route_chain) response: " + bcolors.WHITE + str(response))

                breakpoint()
                response = combined_chain.invoke({"input": "What is Generative AI?"})
                log.info(bcolors.OKGREEN + "response: " + bcolors.WHITE + str(response))

                breakpoint()

            log.info(bcolors.OKGREEN + "(response) content: " + bcolors.WHITE + str(response))
            breakpoint()


        response = llm.invoke(prompt_finetuned)

        if gpt_family: 
            log.info(bcolors.OKGREEN + "(response) content: " + bcolors.WHITE + str(response.content))
            output_dict = output_parser.parse(response.content)
        else:
            output_dict = output_parser.parse(response)
            log.info(bcolors.OKGREEN + "(response) content: " + bcolors.WHITE + str(response))

        log.info(bcolors.OKGREEN + "output_dict: " + bcolors.WHITE + str(output_dict))





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
