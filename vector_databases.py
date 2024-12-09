# Path Handling Libraries
import os
import argparse 
import yaml
from dotenv import load_dotenv, find_dotenv # pip install python-dotenvs
import logging
import sys,os,glob
from tqdm import tqdm 
import getpass

# General Tools 
from gen_tools.tools import bcolors, init_logger,str2bool,get_cmap # , warn_function

# LLM - Related Libraries 
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.evaluation.qa import QAGenerateChain

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import Chroma


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


import numpy as np
import langchain
import openai



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

    embedding = OpenAIEmbeddings()

    sentence1 = "My dog Rover likes to chase squirrels"
    sentence2 = "Fluffy my cat, refuses to eat from a can"
    sentence3 = "The chevy bolt accelerates to 60 mph in 6.7 seconds"

    embedding1 = embedding.embed_query(sentence1)
    embedding2 = embedding.embed_query(sentence2)
    embedding3 = embedding.embed_query(sentence3)

    log.info(bcolors.OKGREEN + "The similarity between sentences one and two is " + bcolors.WHITE + str(np.dot(embedding1, embedding2)))
    log.info(bcolors.OKGREEN + "The similarity between sentences one and three is " + bcolors.WHITE + str(np.dot(embedding1, embedding3)))
    log.info(bcolors.OKGREEN + "The similarity between sentences two and three " + bcolors.WHITE + str(np.dot(embedding3, embedding2)))

    # [Notes] If given duplicated sources of information, the retrieval will return duplicated matches
    # Load PDF
    loaders = [
        # Duplicate documents on purpose - messy data
        PyPDFLoader("data/lecture notes/cs229/cs229-notes1.pdf"),
        PyPDFLoader("data/lecture notes/cs229/cs229-notes2.pdf"),
        PyPDFLoader("data/lecture notes/cs229/cs229-notes3.pdf"),
        PyPDFLoader("data/lecture notes/cs229/cs229-notes2.pdf") # repeated document on purpose
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())


    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 150
    )

    splits = text_splitter.split_documents(docs)
    log.info(bcolors.OKGREEN + "(splits) len: " + bcolors.WHITE + str(len(splits)))


    # Chroma Vector Database: its lightweight and in memory

    persist_directory = 'docs/chroma/'
    os.makedirs(persist_directory, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )

    # Collection len should be equal to the number of generated splits 
    log.info(bcolors.OKGREEN + "(vectordb) len: " + bcolors.WHITE + str(vectordb._collection.count()))
    assert(len(splits)==vectordb._collection.count())


    question = "What are the names and emails of the course TA?"
    docs = vectordb.similarity_search(question,k=3) # Number of retrieved documents is 3 
    log.info(bcolors.OKGREEN + "Retrieved documents: \n" + bcolors.WHITE + str(docs))

    question = "What is the chapter number for the header 'Another algorithm for maximizing ℓ(θ)'"
    docs = vectordb.similarity_search(question,k=3) # Number of retrieved documents is 3 
    log.info(bcolors.OKGREEN + "Retrieved documents: \n" + bcolors.WHITE + str(docs))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_verbosity',  type=int,    default=0, help='Plot results in console in console')
    parser.add_argument('--verbosity',     type=int,    default=1, help='Display messages in console')
    parser.add_argument('--config_filepath', type=str,default="data/config_loaders.yaml", help='Filepath of config file')
    args = parser.parse_args()
    
    # Initialize logger 
    _=init_logger("logger",args.verbosity)
    log = logging.getLogger('logger')
    log.trace("Checking new custom level")


    main(args)
