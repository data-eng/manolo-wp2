from manolo.base.wrappers import version_test
import fastapi; version_test(fastapi)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse