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
from langchain.text_splitter import TokenTextSplitter

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

   
    from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

    chunk_size =26
    chunk_overlap = 4

    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # [Notes] Split based on a single character; default is newline character
    c_splitter_1 = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    c_splitter_2 = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator = ' ' # Character where the splitter is allowed to actually split text 
    )

    # DATA
    # [Notes] Spaces do count as tokens within the chunk size 

    text1 = 'abcdefghijklmnopqrstuvwxyz'
    text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg'
    text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
    text4 = "a b c d e f g h i j k l m n o p q r s t u v w x y z\n\n a b c d e f g h i j k l m n o p q r s t u v w x y z"

    log.info(bcolors.OKGREEN + f"text1: len({len(text1)}): " + bcolors.WHITE + str(text1))
    log.info(bcolors.OKGREEN + f"text2: len({len(text2)}): " + bcolors.WHITE + str(text2))
    log.info(bcolors.OKGREEN + f"text3: len({len(text3)}): " + bcolors.WHITE + str(text3))
    log.info(bcolors.OKGREEN + f"text4: len({len(text4)}): " + bcolors.WHITE + str(text4))

    text1_split_r = r_splitter.split_text(text1)
    text2_split_r =r_splitter.split_text(text2)
    text3_split_r = r_splitter.split_text(text3)

    log.info(bcolors.OKGREEN + "text1_split_r: " + bcolors.WHITE + str(text1_split_r))
    log.info(bcolors.OKGREEN + "text2_split_r: " + bcolors.WHITE + str(text2_split_r))
    log.info(bcolors.OKGREEN + "text3_split_r: " + bcolors.WHITE + str(text3_split_r))

    text3_split_c1 = c_splitter_1.split_text(text3)
    log.info(bcolors.OKGREEN + "text3_split_c1: " + bcolors.WHITE + str(text3_split_c1))
    
    text4_split_c1 = c_splitter_1.split_text(text4)
    log.info(bcolors.OKGREEN + "text4_split_c1: " + bcolors.WHITE + str(text4_split_c1))
    
    text3_split_c2 = c_splitter_2.split_text(text3)
    log.info(bcolors.OKGREEN + "text3_split_c2: " + bcolors.WHITE + str(text3_split_c2))

    text = """As the world grapples with the challenges of climate change, \
    renewable energy emerges as a beacon of hope. Solar and wind power, \
    in particular, are transforming the energy landscape, offering sustainable \
    alternatives to traditional fossil fuels. \n\n  \
    Governments and businesses globally are investing in clean energy \
    initiatives to reduce carbon footprints and mitigate environmental impact. \
    The shift towards renewables not only addresses environmental concerns \
    but also fosters innovation, creating a brighter and more sustainable \
    future for generations to come."""

    log.info(bcolors.OKGREEN + "(text) len: " + bcolors.WHITE + str(len(text)))

    # Character Text Splitter
    c_splitter = CharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=0,
        separator = ' '
    )

    # Recursive Character Text Splitter
    # Works better; more complex, will iterate over meaningfull
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=0, 
        separators=["\n\n", "\n", " ", ""]
    )

    text_split_c = c_splitter.split_text(text)
    log.info(bcolors.OKGREEN + "text_split_c: " + bcolors.WHITE + str(text_split_c))
    log.info(bcolors.OKGREEN + "[text_split_c] len: " + bcolors.WHITE + str([len(chunk) for chunk in text_split_c]))

    text_split_r = r_splitter.split_text(text)
    log.info(bcolors.OKGREEN + "text_split_r: " + bcolors.WHITE + str(text_split_r))
    log.info(bcolors.OKGREEN + "[text_split_r] len: " + bcolors.WHITE + str([len(chunk) for chunk in text_split_r]))


    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=0,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    text_split_r = r_splitter.split_text(text)
    log.info(bcolors.OKGREEN + "text_split_r: " + bcolors.WHITE + str(text_split_r))
    log.info(bcolors.OKGREEN + "[text_split_r] len: " + bcolors.WHITE + str([len(chunk) for chunk in text_split_r]))


    # It is destroying the semantic information of the Text due to Regex Commands
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=0,
        separators=["\n\n", "\n", "\. ", " ", ""]
    )
    text_split_r = r_splitter.split_text(text)
    log.info(bcolors.OKGREEN + "text_split_r: " + bcolors.WHITE + str(text_split_r))
    log.info(bcolors.OKGREEN + "[text_split_r] len: " + bcolors.WHITE + str([len(chunk) for chunk in text_split_r]))

    """
    (?<=...) (Positive Lookbehind): Checks whether an specific pattern appears immediately before the current position in text
        \ is scaping the dot
        . dot 
         the space 
    """

    """
    If the combined size of the current chunk and the next segment (". The shift...) still fits within the chunk_size, the splitter might have chosen to keep it in the same chunk.
    The RecursiveCharacterTextSplitter tries to maximize the chunk size without exceeding the limit. It prioritizes making the chunks as long as possible.
    """

    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=0,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )
    
    text_split_r = r_splitter.split_text(text)
    log.info(bcolors.OKGREEN + "text_split_r: " + bcolors.WHITE + str(text_split_r))
    log.info(bcolors.OKGREEN + "[text_split_r] len: " + bcolors.WHITE + str([len(chunk) for chunk in text_split_r]))


    # ======================================== PART 2 OF DOCUMENT SPLITTING ========================================

    # Load File 
    file = 'data/eBook-How-to-Build-a-Career-in-AI.pdf'
    loader = PyPDFLoader(file_path=file)
    data = loader.load() # list; each of the 

    log.info(bcolors.OKGREEN + "[data] len: " + bcolors.WHITE + str(len(data)))

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    docs = text_splitter.split_documents(data)
    log.info(bcolors.OKGREEN + "[docs] len: " + bcolors.WHITE + str(len(docs)))

    # 1. DIFFERENCE BETWEEN TOKEN & CHARACTER SPLITTING 

    # [Notes] Tokens are the fundamental units a language model processes (e.g., words, subwords, or characters depending on the model).
    # [Notes] Uses a tokenizer (e.g., from OpenAI or Hugging Face) to compute tokens.
    token_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
    text1 = "foo bar bazzyfoo"
    text_split_t = token_splitter.split_text(text1)

    # Character Text Splitter (always by separator = \n\n' by default )
        # [Notes] Operates in character level
    c_splitter = CharacterTextSplitter(
        chunk_size=1,
        chunk_overlap=0,
        separator = ' '
    )

    text_split_c= c_splitter.split_text(text1)

    log.info(bcolors.OKGREEN + "text_split_t: " + bcolors.WHITE + str(text_split_t))
    log.info(bcolors.OKGREEN + "[text_split_t] len: " + bcolors.WHITE + str([len(chunk) for chunk in text_split_t]))

    log.info(bcolors.OKGREEN + "text_split_c: " + bcolors.WHITE + str(text_split_c))
    log.info(bcolors.OKGREEN + "[text_split_c] len: " + bcolors.WHITE + str([len(chunk) for chunk in text_split_c]))

    docs = token_splitter.split_documents(data)
    #  [Notes] docs[0].metadata = data.metadata -> Metadata is preserved in all the chunks


    from langchain.text_splitter import MarkdownHeaderTextSplitter
    # [Notes] Each header works as an index. If you add the same header twice, it is the same index, so it will join both texts after the header
        # ex: ### Section, afterwards ### Section
    markdown_document = """# Title\n\n \
    ## Chapter 1\n\n \
    Hi this is Chapter 1\n\n Hi this is Section 1\n\n \
    ### Section 2 \n\n \
    Hi this is Section 2 \n\n 
    ### Section 3 \n\n \
    Hi this is Section 3 \n\n 
    ## Chapter 2\n\n \
    Hi this is Chapter 2"""

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    md_header_splits = markdown_splitter.split_text(markdown_document)
    log.info(bcolors.OKGREEN + "md_header_splits: " + bcolors.WHITE + str(md_header_splits))



    # [Notes][Additional Info]: https://github.com/hwchase17/chat-langchain-notion
    # [Notes] unzip Export-d3adfe0f-3131-4bf3-8987-a52017fc1bae.zip -d Notion_DB


    # Load the documents using NotionDirectoryLoader
    loader = NotionDirectoryLoader("data/Notion_DB")
    docs = loader.load()

    # Concatenate the content of all pages into a single string
    notion_text = ' '.join([d.page_content for d in docs])

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    md_header_splits = markdown_splitter.split_text(notion_text)
    log.info(bcolors.OKGREEN + "md_header_splits: " + bcolors.WHITE + str(md_header_splits))
    log.info(bcolors.OKGREEN + "md_header_splits [0]: " + bcolors.WHITE + str(md_header_splits[0]))
    
    """
    Es posible que primero divida el texto principal, y luego vaya escaneando cada uno de los documentos que aparecen referenciados
    
    """



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
