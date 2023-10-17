# coding=utf-8
import argparse
import time
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union

import numpy as np
import tiktoken
import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import PolynomialFeatures
from sse_starlette.sse import EventSourceResponse
from starlette.status import HTTP_401_UNAUTHORIZED
from transformers import AutoModel, AutoTokenizer
import os


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", default="16", type=str, help="Model name")
    # args = parser.parse_args()
    # print(args.model_name)
    # # 查看/model下的文件有哪些？
    # print(os.listdir('/model'))
    # # 查看/m3e_base下的文件有哪些？
    # print(os.listdir('/m3e_base'))
    print('ioio')
