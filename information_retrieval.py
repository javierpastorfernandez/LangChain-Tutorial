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
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain.evaluation.qa import QAGenerateChain

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains.query_constructor.base import AttributeInfo


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
    # TODO: HARDCODED
    execution_stages = ["retieval_comparison"]

    log.info(bcolors.OKGREEN + "execution_stages: " + bcolors.WHITE + str(execution_stages))

    _ = load_dotenv(find_dotenv()) # read local .env file

    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

    if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your HUGGINFACEHUB API key: ")

    openai.api_key = os.environ['OPENAI_API_KEY']
    log.info(bcolors.OKGREEN + "open_ai key: " + bcolors.WHITE + str(openai.api_key))
    log.info(bcolors.OKGREEN + "HUGGINGFACEHUB_API_TOKEN: " + bcolors.WHITE + str(os.environ["HUGGINGFACEHUB_API_TOKEN"]))

    # Chroma Vector Database: its lightweight and in memory
    persist_directory = 'docs/chroma/'
    os.makedirs(persist_directory, exist_ok=True)

    embedding = OpenAIEmbeddings()

    # Same but without the documents attribute
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )

    log.info(bcolors.OKGREEN + "(vectordb) len: " + bcolors.WHITE + str(vectordb._collection.count()))

    # 1. MAXIMUM MARGINAL RELEVANCE
    if "mmr" in execution_stages:

        texts = [
            """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
            """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
            """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
        ]
        smalldb = Chroma.from_texts(texts, embedding=embedding)
        log.info(bcolors.OKGREEN + "(smalldb) len: " + bcolors.WHITE + str(smalldb._collection.count()))

        question = "Tell me about all-white mushrooms with large fruiting bodies"

        docs = smalldb.similarity_search(question, k=2)
        log.info(bcolors.OKGREEN + "[similarity_search] Retrieved documents: \n" + bcolors.WHITE + str(docs))


        docs = smalldb.max_marginal_relevance_search(question,k=2, fetch_k=3)
        log.info(bcolors.OKGREEN + "[max_marginal_relevance_search] Retrieved documents: \n" + bcolors.WHITE + str(docs))

        # [Notes] Difference is Max Marginal Relevance wont retrieve the same source twice
        question = "what did they say about matlab?"
        docs = vectordb.max_marginal_relevance_search(question,k=3) # Number of retrieved documents is 3 
        log.info(bcolors.OKGREEN + "[max_marginal_relevance_search] Retrieved documents: \n" + bcolors.WHITE + str(docs))

    # 2. LLM - AIDED RETRIEVAL 
    if "llm_aided" in execution_stages:
        question = "what did they say about Support Vector Machines in the third lecture?"
        docs = vectordb.similarity_search(
            question,
            k=3,
            filter={"source":"data/lecture notes/cs229/cs229-notes3.pdf"}
        )
        for d in docs:
            log.info(bcolors.OKGREEN + "[docs] metadata: " + bcolors.WHITE + str(d.metadata))
        
        """
        docs = vectordb.similarity_search(
            question,
            k=3,
            filter={"source":"data/lecture notes/cs229/cs229-notes2.pdf"}
        )
        for d in docs:
            log.info(bcolors.OKGREEN + "[docs] metadata: " + bcolors.WHITE + str(d.metadata))
        """
                    
        # [Notes] Prepares the Retriever to work with these two filters 
        metadata_field_info = [
            AttributeInfo(
                name="source",
                description="The lecture the chunk is from, should be one of `data/lecture notes/cs229/cs229-notes3.pdf`, `data/lecture notes/cs229/cs229-notes1.pdf`,`data/lecture notes/cs229/cs229-notes2.pdf`",
                type="string",
            ),
            AttributeInfo(
                name="page",
                description="The page from the lecture",
                type="integer",
            ),
        ]

        document_content_description = "Lecture notes"
        llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)

        retriever = SelfQueryRetriever.from_llm(
            llm,
            vectordb,
            document_content_description,
            metadata_field_info,
            verbose=True
        )

        # [Notes] This wont activate the filtering step of the retriever
        question = "what did they say about Support Vector Machines in the forth lecture?"
        # This example only specifies a filter
        docs = retriever.invoke(question)
        for d in docs:
            log.info(bcolors.OKGREEN + "[docs] metadata: " + bcolors.WHITE + str(d.metadata))


        # [Notes] This activates the filtering step of the retriever!
        question = "what did they say about Support Vector Machines?. Source should be `data/lecture notes/cs229/cs229-notes2.pdf`"
        # This example only specifies a filter
        docs = retriever.invoke(question)
        for d in docs:
            log.info(bcolors.OKGREEN + "[docs] metadata: " + bcolors.WHITE + str(d.metadata))
        
        # [Notes] This activates the filtering step of the retriever!
        question = "what did they say about Support Vector Machines?. Source should be the first lecture"
        # This example only specifies a filter
        docs = retriever.invoke(question)
        for d in docs:
            log.info(bcolors.OKGREEN + "[docs] metadata: " + bcolors.WHITE + str(d.metadata))
        

    # 2. RETRIEVAL BY COMPARISON 
    if "retieval_comparison" in execution_stages:
        def pretty_print_docs(docs):
            print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


        # Wrap our vectorstore
        llm = OpenAI(temperature=0)
        compressor = LLMChainExtractor.from_llm(llm)

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vectordb.as_retriever()
        )
        
        # [Notes] Using semantic serach
        question = "what did they say about Support Vector Machines?"
        compressed_docs = compression_retriever.invoke(question)
        pretty_print_docs(compressed_docs)

        # [Notes] Using Maximum Marginal Relevance
        # [Warning] Deprecated 
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vectordb.as_retriever(search_type = "mmr")
        )
        compressed_docs = compression_retriever.invoke(question)
        pretty_print_docs(compressed_docs)
     


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
