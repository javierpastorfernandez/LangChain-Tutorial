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
from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate

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
    execution_stages = ["chatbot"]

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


    llm = ChatOpenAI(model_name=llm_model, temperature=0)
    question = "What are major topics for this class?"

    """
    index = VectorstoreIndexCreator(
        vectorstore_cls=DocArrayInMemorySearch,
        embedding=embedding_model
    ).from_loaders([loader])

    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=index.vectorstore.as_retriever(), 
        verbose=True,
        chain_type_kwargs = {
            "document_separator": "<<<<>>>>>"
        })
    """
    langchain.debug = True


    if "qa_chain" in execution_stages:
        # [WARNING] [NOTES] The content from the RetrievalQA & the similarity search are different 
        docs = vectordb.similarity_search(question,k=3)
        log.info(bcolors.OKGREEN + "[similarity_search] Retrieved documents: \n" + bcolors.WHITE + str(docs))
        # [Notes] Inside the Retrieval QA is building a prompt with the following structure
            # "System: Use the following pieces of context to answer the user's question. \nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\nCS229 Lecture notes\

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            chain_type="stuff", # default
            retriever=vectordb.as_retriever(),
            # verbose = True,
        )
        response = qa_chain.invoke({"query": question})
        log.info(bcolors.OKGREEN + "(qa_chain) response: " + bcolors.WHITE + str(response["result"]))


        # Build prompt

        question = "Is probability a class topic?"

        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
        {context}
        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # Run chain
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        response = qa_chain.invoke({"query": question}) # dict_keys(['query', 'result', 'source_documents'])
        log.info(bcolors.OKGREEN + "(qa_chain) response: " + bcolors.WHITE + str(response["result"]))
        log.info(bcolors.OKGREEN + f"(qa_chain) (response) sources [{len(response['source_documents'])}]:\n" + bcolors.WHITE + str(response["source_documents"]))

        # stuffs: to fill tighly
        # RetrievalQA with Map Reduce & Refined Chain

    
    # [Warning] It retrieves Errors due to LangChain installation 
    if "map_reduced" in execution_stages:
        
        qa_chain_mr = RetrievalQA.from_chain_type(
            llm,
            retriever=vectordb.as_retriever(),
            chain_type="map_reduce"
        )

        response = qa_chain_mr({"query": question})

        log.info(bcolors.OKGREEN + "(qa_chain) response: " + bcolors.WHITE + str(response["result"]))
        log.info(bcolors.OKGREEN + f"(qa_chain) (response) sources [{len(response['source_documents'])}]:\n" + bcolors.WHITE + str(response["source_documents"]))
    
    if "refine" in execution_stages:
        # [Notes] Iterativo y secuencial, realiza iteraciones de la misma cadena, y hace un append de las respuestas anteriores en "existing answer"
        
        """ Ejemplo:
            "context_str": "CS229 Lecture notes\nAndrew Ng\nPart IV\nGenerative Learning algorithms\nSo far, we’ve mainly been talking about learning algorithms that model\np(y|x; θ), the conditional distribution of y given x. For instance, logistic\nregression modeled p(y|x; θ) as hθ(x) = g(θT x) where g is the sigmoid func-\ntion. In these notes, we’ll talk about a diﬀerent type of learning algo rithm.\nConsider a classiﬁcation problem in which we want to learn to distinguish\nbetween elephants ( y = 1) and dogs ( y = 0), based on some features of\nan animal. Given a training set, an algorithm like logistic regression or\nthe perceptron algorithm (basically) tries to ﬁnd a straight line—that is, a\ndecision boundary—that separates the elephants and dogs. Then , to classify\na new animal as either an elephant or a dog, it checks on which side of t he\ndecision boundary it falls, and makes its prediction accordingly.\nHere’s a diﬀerent approach. First, looking at elephants, we can build a\nmodel of what elephants look like. Then, looking at dogs, we can build a\nseparate model of what dogs look like. Finally, to classify a new animal,we\ncan match the new animal against the elephant model, and match it ag ainst\nthe dog model, to see whether the new animal looks more like the eleph ants\nor more like the dogs we had seen in the training set.\nAlgorithms that try to learn p(y|x) directly (such as logistic regression),\nor algorithms that try to learn mappings directly from the space of in puts X",
            "existing_answer": "The major topics covered in this class, based on the context provided, include:\n1. Supervised learning\n2. Dataset analysis\n3. Regression analysis\n4. Predictive modeling\n5. Machine learning algorithms\n6. Feature selection\n7. Model evaluation and validation\n8. Predicting housing prices based on living area\n9. Application of machine learning in real-world scenarios",
            "question": "What are major topics for this class?"
        """

        qa_chain_mr = RetrievalQA.from_chain_type(
            llm,
            retriever=vectordb.as_retriever(),
            chain_type="refine"
        )
        response = qa_chain_mr({"query": question})

        log.info(bcolors.OKGREEN + "(qa_chain) response: " + bcolors.WHITE + str(response["result"]))


    # [Notes] The concept of memory needs to be introduced to provide ChatBot capabilities the LLM & QAChain
    if "chatbot" in execution_stages:
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectordb.as_retriever()
        )

        question = "Is probability a class topic?"
        response = qa_chain({"query": question})
        log.info(bcolors.OKGREEN + "(qa_chain) response: " + bcolors.WHITE + str(response["result"]))

        question = "why are those prerequesites needed?"
        response = qa_chain({"query": question})
        log.info(bcolors.OKGREEN + "(qa_chain) response: " + bcolors.WHITE + str(response["result"]))




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
