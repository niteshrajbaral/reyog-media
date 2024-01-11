import os
import sys
import langchain
# import openai

import constants

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = constants.APIKEY

query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

loader = TextLoader("./data/data.txt")
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query))