import os
import openai
import argparse 
import yaml
from dotenv import load_dotenv, find_dotenv # pip install python-dotenvs
import logging
import sys,os,glob
from tqdm import tqdm 
from gen_tools.tools import bcolors, init_logger,str2bool,get_cmap # , warn_function

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

import getpass

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader

#  ¡[Notes] Use this instead to avoid Warnings! 
# from langchain.indexes import VectorstoreIndexCreator
# from langchain_community.vectorstores import DocArrayInMemorySearch
# from langchain_community.document_loaders import PyPDFLoader
#  =========================================================================================

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.evaluation.qa import QAGenerateChain
import numpy as np

import langchain
from langchain.evaluation.qa import QAEvalChain


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
    # llm_model="gpt-3.5-turbo-0301" # DEPRECATED:  TODO: HARDCODED

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


    # Load File 
    file = 'data/eBook-How-to-Build-a-Career-in-AI.pdf'
    loader = PyPDFLoader(file_path=file)
    data = loader.load() # list; each of the items is a page of the original PDF 

    # TODO: HARDCODED: Reduce input size
    data = data[0:2]

    # Create an embedding instance
    embedding_model = OpenAIEmbeddings()  # or another embedding model that you have access to

    # Pass the embedding model to VectorstoreIndexCreator
    index = VectorstoreIndexCreator(
        vectorstore_cls=DocArrayInMemorySearch,
        embedding=embedding_model
    ).from_loaders([loader])

    llm = ChatOpenAI(temperature = 0.0, model=llm_model)
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=index.vectorstore.as_retriever(), 
        verbose=True,
        chain_type_kwargs = {
            "document_separator": "<<<<>>>>>"
        })

    """
    data[10]: This part of the document discusses th importance of continuous learning and developing the habit of learning.
    data[8]:  The second page explores the different skills you need to start a career in AI and data science. So the first one we can ask is a simple question Is machine learning foundations the most important skill?
    """

    examples = [
        {   "query": "Is machine learning foundations the most important skill?",
            "answer": "Yes"
        },
        {   "query": "What are Python frameworks you need to learn ?",
            "answer": "Tensorflow and PyTorch"
        }]
    
    # QA generation chain, basically a LLM chain that creates a question-answer pair from each document. 
    qa_llm_chain = QAGenerateChain.from_llm(llm)
    # output_parser = StructuredOutputParser

    input_data = [{"doc": t} for t in data[:5]]
    query_n_responses = [qa_llm_chain.invoke(dict_) for dict_ in tqdm(input_data)]
    # examples = examples + [example["qa_pairs"] for example in query_n_responses]

    # TODO: HARDCODED
    examples = [example["qa_pairs"] for example in query_n_responses]
    # new_examples = example_gen_chain.apply_and_parse([{“doc”: data[:5]}])



    langchain.debug = True
    query_index = np.min([2, len(query_n_responses)-1])
    response = qa.invoke(examples[query_index]["query"])
    log.trace(bcolors.OKGREEN + "response: " + bcolors.WHITE + str(response))
    log.trace(bcolors.OKGREEN + "(response) Ground Truth: " + bcolors.WHITE + str(examples[query_index]["answer"]))

    # You can pass both Queries & The whole dictionary, because the Chain will arrange correct input to the LLM 
    # queries = [example["query"] for example in examples]
    # predictions = qa.batch(queries) # Output Dictionary Contains (Queries & Results)
    predictions = qa.batch(examples) # Output Dictionary Contains (Queries & Results & Answers)
    eval_chain = QAEvalChain.from_llm(llm)

    # This is raising errors 
    graded_outputs = eval_chain.evaluate(examples, predictions)


    for idx_i, eg in enumerate(examples):
        log.trace(bcolors.OKGREEN + "Example: " + bcolors.WHITE + str(idx_i))
        log.trace(bcolors.OKGREEN + "Question: " + bcolors.WHITE + str(predictions[idx_i]['query']))
        log.trace(bcolors.OKGREEN + "Real Answer: " + bcolors.WHITE + str(predictions[idx_i]['answer']))
        log.trace(bcolors.OKGREEN + "Predicted Answer: " + bcolors.WHITE + str(predictions[idx_i]['result']))
        log.trace(bcolors.OKGREEN + "Predicted Grade: " + bcolors.WHITE + str(graded_outputs[idx_i]['text']))



        breakpoint()

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
