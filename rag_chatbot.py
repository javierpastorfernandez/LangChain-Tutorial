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
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain.prompts import PromptTemplate

import numpy as np
import langchain
import openai

import panel as pn
import param


def load_db(file, llm_name, chain_type, k):

    """
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

    """
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa



class cbfs(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query  = param.String("")
    db_response = param.List([])
    
    def __init__(self, llm_model,  file_input, button_load, inp, **params):
        super().__init__(**params)  # Initialize with the remaining params
        self.llm_model = llm_model  # Explicitly assign the llm_model
        self.file_input = file_input
        self.button_load = button_load
        self.inp = inp
        self.panels = []
        self.loaded_file = "data/lecture notes/cs229/cs229-notes1.pdf"
        self.qa = load_db(self.loaded_file, self.llm_model, "stuff", 4)

    def call_load_db(self, count):
        if count == 0 or self.file_input.value is None:  # init or no file specified :
            return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")
        else:
            self.file_input.save("temp.pdf")  # local copy
            self.loaded_file = self.file_input.filename
            self.button_load.button_style="outline"
            self.qa = load_db("temp.pdf", self.llm_model,  "stuff", 4)
            self.button_load.button_style="solid"
        self.clr_history()
        return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")

    def convchain(self, query):
        if not query:
            return pn.WidgetBox(pn.Row('User:', pn.pane.Markdown("", width=600)), scroll=True)
        result = self.qa.invoke({"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer'] 
        self.panels.extend([
            pn.Row('User:', pn.pane.Markdown(query, width=600)),
            pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=600, styles={'background-color': '#F6F6F6'}))
        ])
        self.inp.value = ''  #clears loading indicator when cleared
        return pn.WidgetBox(*self.panels,scroll=True)

    @param.depends('db_query ', )
    def get_lquest(self):
        if not self.db_query :
            return pn.Column(
                pn.Row(pn.pane.Markdown(f"Last question to DB:", styles={'background-color': '#F6F6F6'})),
                pn.Row(pn.pane.Str("no DB accesses so far"))
            )
        return pn.Column(
            pn.Row(pn.pane.Markdown(f"DB query:", styles={'background-color': '#F6F6F6'})),
            pn.pane.Str(self.db_query )
        )

    @param.depends('db_response', )
    def get_sources(self):
        if not self.db_response:
            return 
        rlist=[pn.Row(pn.pane.Markdown(f"Result of DB lookup:", styles={'background-color': '#F6F6F6'}))]
        for doc in self.db_response:
            rlist.append(pn.Row(pn.pane.Str(doc)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    @param.depends('convchain', 'clr_history') 
    def get_chats(self):
        if not self.chat_history:
            return pn.WidgetBox(pn.Row(pn.pane.Str("No History Yet")), width=600, scroll=True)
        rlist=[pn.Row(pn.pane.Markdown(f"Current Chat History variable", styles={'background-color': '#F6F6F6'}))]
        for exchange in self.chat_history:
            rlist.append(pn.Row(pn.pane.Str(exchange)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    def clr_history(self,count=0):
        self.chat_history = []
        return


# ssh -p 22 -N -f -L localhost:43131:localhost:43131 javpasto@10.222.194.205
# http://localhost:43131

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

    file_input = pn.widgets.FileInput(accept='.pdf')
    button_load = pn.widgets.Button(name="Load DB", button_type='primary')
    inp = pn.widgets.TextInput( placeholder='Enter text here…')
    cb = cbfs(llm_model = llm_model,  file_input = file_input, button_load = button_load, inp = inp)


    button_clearhistory = pn.widgets.Button(name="Clear History", button_type='warning')
    button_clearhistory.on_click(cb.clr_history)

    bound_button_load = pn.bind(cb.call_load_db, button_load.param.clicks)
    conversation = pn.bind(cb.convchain, inp) 

    jpg_pane = pn.pane.Image( './img/convchain.jpg')

    tab1 = pn.Column(
        pn.Row(inp),
        pn.layout.Divider(),
        pn.panel(conversation,  loading_indicator=True, height=300),
        pn.layout.Divider(),
    )
    tab2= pn.Column(
        pn.panel(cb.get_lquest),
        pn.layout.Divider(),
        pn.panel(cb.get_sources ),
    )
    tab3= pn.Column(
        pn.panel(cb.get_chats),
        pn.layout.Divider(),
    )
    tab4=pn.Column(
        pn.Row( file_input, button_load, bound_button_load),
        pn.Row( button_clearhistory, pn.pane.Markdown("Clears chat history. Can use to start a new topic" )),
        pn.layout.Divider(),
        pn.Row(jpg_pane.clone(width=400))
    )
    dashboard = pn.Column(
        pn.Row(pn.pane.Markdown('# ChatWithYourData_Bot')),
        pn.Tabs(('Conversation', tab1), ('Database', tab2), ('Chat History', tab3),('Configure', tab4))
    )



    dashboard.servable()
    pn.serve(dashboard)



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
