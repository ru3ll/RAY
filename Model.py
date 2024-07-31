import os
from pathlib import PurePath
from threading import Thread
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
try:
    from langchain_chroma import Chroma
except:
    from langchain.vectorstores import Chroma

from langchain.embeddings import SentenceTransformerEmbeddings
import uuid
import onnxruntime as ort

from transformers import AutoTokenizer
import requests
try:
    # Try importing wrapper class
    from include.llm import ORTModelEval as ORTModelForCausalLM
except ImportError:
    print("[RyzenAI::W] Importing from optimum.onnxruntime")
    from optimum.onnxruntime.modeling_decoder import ORTModelForCausalLM

from include.tasks import Tasks
from transformers import TextIteratorStreamer
from include.logger import LINE_SEPARATER, RyzenAILogger
from pathlib import PurePath

onnx_model_dir = "C:/Users/brian/OneDrive/Desktop/llama3-8b-onnx/quant"
model_dir_base = PurePath(onnx_model_dir).parts[-1]
# Enable Logging
logger = RyzenAILogger(model_dir_base)
# Start logging
logger.start()
# ORT Logging level
# Logging severity -> 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal
ort.set_default_logger_severity(3)
import json
import os
from email.message import EmailMessage
import ssl
import smtplib
from dotenv import load_dotenv
import logging
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

from langchain_core.prompts import SystemMessagePromptTemplate
from agent_tools import get_current_weather, get_random_joke,install_package, send_mail, conversational_response

DEFAULT_SYSTEM_TEMPLATE_FOR_MEMORY = """
    Your name is Ray. You are a helpful assistant. Respond honestly to the users prompt. Do not make references to the past conversations. Keep the conversation as natural as possible.
    You have been given access to a vector database that will return relevant pieces of the conversation based on the user prompt
    The relevant messages will be a list of at most 4 messages from the past. You also have access to the sliding window of conversational history

    Relevant pieces of information from the users prompt:
    {relevant_pieces}
    
    A history of the current conversation is given below. NOTE, this contains only the past few messages:

    {history}

    """

DEFAULT_RAG_TEMPLATE = """
You are a helpful assistant capable of reading through pdfs. A vector database will provide a list of relevant pieces from a user supplied document. This list is given below

{relevant_pieces}

Use it to respond to the user's prompt as comprehensively as possible.
"""

DEFAULT_SYSTEM_TEMPLATE = """Your name is Ray. You are a helpful assistant. You have access to the following tools:

{tools}
    

You must always select one of the above tools and respond with ONLY a JSON object matching the following schema:

{{
"tool": <name of the selected tool>,
"tool_input": <parameters for the selected tool, matching the tool's JSON schema>
}}
"""  # noqa: E501

tools = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "send_mail",
        "description": "Send an email to a specified email address",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "The email address of the recipient e.g. bmacharia474@gmail.com",
                },
                "subject": {
                    "type": "string",
                    "description": "The email subject",
                },
                "body": {
                    "type": "string",
                    "description": "The contents of the email also known as the body of an email",
                },
                
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "get_random_joke",
        "description": "Get a random joke",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },

    {
        "name": "conversational_response",
        "description": (
            "Respond conversationally if no other tools should be called for a given query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "Conversational response to the user.",
                },
            },
            "required": ["response"],
        },
    },
    {
        "name": "install_package",
        "description": (
            "Install a given package using winget"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "package": {
                    "type": "string",
                    "description": "package to install",
                },
            },
            "required": ["package"],
        },
    },
]



class Model:
    def __init__(self, target = "cpu", impl = "eager"):
        self.impl = impl
        self.agent_message = SystemMessagePromptTemplate.from_template(
            DEFAULT_SYSTEM_TEMPLATE
        ).format(
            tools=json.dumps(tools, indent=2)
        )
        self.history_db = "history"
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
        if os.path.exists(self.history_db): #check whether the vector database exists
            self.db = Chroma(persist_directory=self.history_db,
                             embedding_function=self.embedding_function)
        else: # create a new database with a pseudo entry
            pseudo_content = {"role" : "user", "content" : "Hello"}
            self.db = Chroma.from_texts(
                texts=[pseudo_content["content"]],
                metadatas=[pseudo_content],
                ids=["id0"],
                embedding=self.embedding_function,
                persist_directory=self.history_db
            )
        self.retriever = self.db.as_retriever()
        self.model_id  = "D:/llama3-8b-instruct"  #local path to model
        self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                token=True,
            )
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.streamer = TextIteratorStreamer(tokenizer=self.tokenizer, skip_prompt = True, skip_special_tokens=True)
        self.history = []
        print("loading model")

        if self.impl == "onnx":
            print("loading onnx model")
            self.onnx_model_dir = "C:/Users/brian/OneDrive/Desktop/llama3-8b-onnx/quant"
            self.target = target
            # Create session option
            self.sess_options = ort.SessionOptions()

            

            if self.target == "cpu":
                self.ep = "CPUExecutionProvider"
            elif self.target == "aie":
                self.ep = "VitisAIExecutionProvider"
            else:
                self.ep = "DmlExecutionProvider"

            self.model_args = {}
            self.model_args["use_cache"] = True
            self.model_args["use_io_binding"] = True
            self.model_args["trust_remote_code"] = False
            self.model_args["provider"] = self.ep
            self.model_args["session_options"] = self.sess_options

            if self.ep == "VitisAIExecutionProvider":
                self.model_args["provider_options"] = {
                    "config_file": "vaip_config_transformers.json"
                }
            elif self.ep == "DmlExecutionProvider":
                self.model_args["provider_options"] = {
                    "device_id": "0"
                }

            # Model
            self.model = ORTModelForCausalLM.from_pretrained(onnx_model_dir, **self.model_args)
        else:
            print("loading on eager mode")
            import torch
            from ryzenai_llm_engine import RyzenAILLMEngine, TransformConfig
            from ryzenai_llm_quantizer import QuantConfig, RyzenAILLMQuantizer
            self.ckpt = "C:/Users/brian/OneDrive/Desktop/Ray-2/transformers/models/llm/quantized_models/quantized_llama3-8b-instruct_w4_g128_awq.pth"
            self.model = torch.load(self.ckpt)
            transform_config = TransformConfig(
                flash_attention_plus= False,
                fast_mlp= False,
                fast_attention= False,
                precision= "w4abf16",
                model_name= "llama3-8b-instruct",
                target= "aie",
                w_bit= 4,
                group_size= 128,
                profilegemm= False,
                profile_layer= False,
                mhaops= "all",
            )

            self.model = RyzenAILLMEngine.transform(self.model, transform_config)
            self.model = self.model.to(torch.bfloat16)
            self.model.eval()
            print(self.model)
            print(f"model.mode_name: {self.model.model_name}")

        

    def decode(self, prompt, max_new_tokens = 1024):
        docs = self.retriever.invoke(prompt)
        docs = [doc.metadata for doc in docs]
        memory_message = SystemMessagePromptTemplate.from_template(
            DEFAULT_SYSTEM_TEMPLATE_FOR_MEMORY
        ).format(
            relevant_pieces = json.dumps(docs),
            history=json.dumps(self.history if len(self.history) < 8 else self.history[:-8]) # pass in the last 4 messages + responses from the history
        )
        formatted_prompt = {"role": "user", "content": prompt}
        messages = [
            {"role": "system", "content": memory_message.content},
            formatted_prompt,
        ]
        self.history.append(formatted_prompt)
        self.db.add_texts(
            texts=[formatted_prompt["content"]],
            metadatas=[formatted_prompt],
            ids = [str(uuid.uuid4())]
        )
        print("starting generation")
        inputs= self.tokenizer.apply_chat_template(messages, tokenize = True,add_generation_prompt=True,return_tensors="pt")
        # print(inputs)
        thread = Thread(target=self.model.generate, kwargs={"input_ids": inputs,
                            "streamer": self.streamer, "max_new_tokens": max_new_tokens,
                            "pad_token_id":self.tokenizer.eos_token_id})
        thread.start()
        message = ""
        for new_text in self.streamer:
            message += new_text
            yield new_text
        formatted_message = {"role": "ai", "content":message}
        self.history.append(formatted_message)
        self.db.add_texts(
            texts=[formatted_message["content"]],
            metadatas=[formatted_prompt],
            ids = [str(uuid.uuid4())]
        )


    def agent(self, prompt):
        formatted_prompt = {"role": "user", "content": prompt}
        messages = [
            {"role": "system", "content": self.agent_message.content},
            formatted_prompt,
        ]
        inputs= self.tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt", padding=True)
        # print(inputs)
        thread = Thread(target=self.model.generate, kwargs={"input_ids": inputs,
                            "streamer": self.streamer, "max_new_tokens": 128,
                            "pad_token_id":self.tokenizer.eos_token_id})
        thread.start()
        parser = ""
        for new_text in self.streamer:
            parser+=new_text
        
        # # thread.join()
        print(parser)
        try:
            tool = json.loads(parser.split("\"tool\": ")[1].split(",")[0])
            args = json.loads("{" +parser.split("\"tool\": ")[1].split("\"tool_input\": {")[1].split("}")[0] + "}")
            parser = {"tool" : tool, "tool_input" : args}
            print(parser)
            # parser = json.loads(parser.split("{", 1)[1].rsplit("}", 0)[0].replace("\n", ""))
            if parser["tool"] == "get_current_weather":
                # print("calling function {tool}")
                return get_current_weather(parser["tool_input"]["location"], parser["tool_input"]["unit"])
            elif parser["tool"] == "get_random_joke":
                # print("calling function {tool}")
                return get_random_joke()
            elif parser["tool"] == "send_mail":
                # print("calling function {tool}")
                return send_mail(parser["tool_input"]["to"], parser["tool_input"]["subject"], parser["tool_input"]["body"])
            elif parser["tool"] == "conversational_response":
                # print("calling function {tool}")
                return conversational_response(parser["tool_input"]["response"])
            elif parser["tool"] == "install_package":
                # print("calling function {tool}")
                return install_package(parser["tool_input"]["package"])
        except IndexError:
            return parser        
    
    
    def RAG(self, filename, prompt, max_new_tokens = 512):
        file = f'./uploads/{filename}'
        loader = PyPDFLoader(file)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        vector_db = Chroma.from_documents(texts,embedding=self.embedding_function)
        retreiver = vector_db.as_retriever()
        docs = retreiver.invoke(prompt)
        print(docs)
        docs = [doc.page_content for doc in docs]
        print(docs)
        memory_message = SystemMessagePromptTemplate.from_template(
            DEFAULT_RAG_TEMPLATE
        ).format(
            relevant_pieces = json.dumps(docs)
        )
        formatted_prompt = {"role": "user", "content": prompt}
        messages = [
            {"role": "system", "content": memory_message.content},
            formatted_prompt,
        ]
        inputs= self.tokenizer.apply_chat_template(messages, tokenize = True,add_generation_prompt=True,return_tensors="pt")
        # print(inputs)
        thread = Thread(target=self.model.generate, kwargs={"input_ids": inputs,
                            "streamer": self.streamer, "max_new_tokens": max_new_tokens,
                            "pad_token_id":self.tokenizer.eos_token_id})
        thread.start()
        message = ""
        for new_text in self.streamer:
            message += new_text
            yield new_text
        