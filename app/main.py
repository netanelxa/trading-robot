from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI

app = FastAPI()


#@app.get("/")
#def read_root():
#    return {"Hello": "World"}

app.mount("/", StaticFiles(directory="./app//static",html = True), name="static")

