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
import numpy as np

import langchain
import openai

# Dataloaders
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import NotionDirectoryLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders import WebBaseLoader

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

    # 1.0 - Load PDF 
    if "pdf" in execution_stages:
        file = 'data/eBook-How-to-Build-a-Career-in-AI.pdf'
        loader = PyPDFLoader(file_path=file)
        data_pdf = loader.load() # list; each of the items is a page of the original PDF 
        log.info(bcolors.OKGREEN + "(pdf) number pages: " + bcolors.WHITE + str(len(data_pdf)))
        log.info(bcolors.OKGREEN + "(pdf) [page 0] metadata: " + bcolors.WHITE + str(data_pdf[0].metadata))

    # 2.0 - Load CSV
    if "csv" in execution_stages:
        loader = CSVLoader(file_path="data/earthquakes.csv")
        data_csv = loader.load()
        log.info(bcolors.OKGREEN + "(pdf) [page 0] metadata: " + bcolors.WHITE + str(data_csv[0].metadata))
        # List
            # dict():
                # metadata
                # page_content (str)
    
    # 3.0 - Load Excel (XLS)
    if "excel" in execution_stages:
        loader = UnstructuredExcelLoader("data/orders_with_nulls.xlsx", mode="elements")
        data_excel = loader.load()
        log.info(bcolors.OKGREEN + "(data_excel) [page 0] metadata: " + bcolors.WHITE + str(data_excel[0].metadata))
        log.info(bcolors.OKGREEN + "(data_excel) [page 0]: " + bcolors.WHITE + str(data_excel[0]))

    # 4.0 Load Microsoft Word Files
    if "word" in execution_stages:
            # Working with docx files
            # Not working with doc files
            # [Notes] Reference: https://www.folderit.com/blog/difference-between-doc-and-docx-which-should-you-use/
            
        loader = Docx2txtLoader("data/sample_docx.docx")
        data_doc = loader.load()
        log.info(bcolors.OKGREEN + "(data_doc) [page 0] page_content: " + bcolors.WHITE + str(data_doc[0].page_content))
        log.info(bcolors.OKGREEN + "(data_doc) [page 0] metadata: " + bcolors.WHITE + str(data_doc[0].metadata))

    # 5.0 Load Youtube Metrics 
    if "youtube" in execution_stages:
        url = "https://www.youtube.com/watch?v=5p248yoa3oE"
        save_dir = "data/youtube/"
        loader = GenericLoader(
            YoutubeAudioLoader([url],save_dir),
            OpenAIWhisperParser()
        )
        data_video = loader.load()
        log.info(bcolors.OKGREEN + "(data_video) [page 0] page_content: " + bcolors.WHITE + str(data_video[0].page_content[0:500]))

    # 6.0 - Loading HTML Pages 
    if "html" in execution_stages:
        loader = WebBaseLoader("https://github.com/youssefHosni/Getting-Started-with-Generative-AI")
        data_html = loader.load()

        log.info(bcolors.OKGREEN + "(data_html) [page 0] metadata: " + bcolors.WHITE + str(data_html[0].metadata))
        log.info(bcolors.OKGREEN + "(data_html) [page 0] page_content: " + bcolors.WHITE + str(data_html[0].page_content[6000:7000]))


    # 8.0 - Loading Notion Pages 
    if "notion" in execution_stages:
        
        filepath = "data/notion"
        assert(os.path.isdir(filepath))
        breakpoint()
        loader = NotionDirectoryLoader(filepath)
        data_notion = loader.load()

        log.info(bcolors.OKGREEN + "(data_notion) [page 0] metadata: " + bcolors.WHITE + str(data_notion[0].metadata))
        log.info(bcolors.OKGREEN + "(data_notion) [page 0] page_content: " + bcolors.WHITE + str(data_notion[0].page_content[0:1000]))


    # 9.0 - General Document Loading 
    if "general" in execution_stages:
        """
        DOCUMENT_MAP = {
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
            ".py": TextLoader,
            ".pdf": UnstructuredFileLoader,
            ".csv": CSVLoader,
            ".xls": UnstructuredExcelLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".docx": Docx2txtLoader,
            ".doc": Docx2txtLoader,
            ".json": JSONLoader
            }
        """


        DOCUMENT_MAP = {
            ".csv": CSVLoader,
            ".xls": UnstructuredExcelLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".docx": Docx2txtLoader,
            ".doc": Docx2txtLoader,
            }

        docx_file_path = "data/sample_docx.docx"
        file_extension = os.path.splitext(docx_file_path)[1]
        loader_class = DOCUMENT_MAP.get(file_extension)

        loader = loader_class(docx_file_path)
        data = loader.load()

        log.info(bcolors.OKGREEN + "(data) [page 0] page_content: " + bcolors.WHITE + str(data[0].page_content))
        log.info(bcolors.OKGREEN + "(data) [page 0] metadata: " + bcolors.WHITE + str(data[0].metadata))


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
