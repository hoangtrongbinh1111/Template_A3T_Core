from threading import Thread
import os
import pickle
import asyncio
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import socketio
import json
import uvicorn
from train import train
from test import test
from inference import infer
from tool import *

TIMEOUT_LIMIT = 60 * 60 * 24  # Example timeout duration
SOCKET_BACKEND_URL = "http://url-phishing-service:12007"
PORT = 12009

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sio = socketio.AsyncClient(logger=True, engineio_logger=True)

def run_async(func, *args):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(func(*args))
    loop.close()

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to AI Core"}


@app.on_event("startup")
async def startup():
    await sio.connect(SOCKET_BACKEND_URL)


@sio.event
async def connect():
    print("connection established")


async def start_training(data):
    try:
        async def func_with_timeout():
            async for response in train(
                data["data_dir"],
                data["learning_rate"],
                data["epochs"],
                data["batch_size"],
                data["val_size"],
                data["embed_size"],
                data["num_neurons"],
                data["num_layers"],
                data["backbone"],
                data["model_type"],
                data["labId"]
            ):
                await sio.emit(
                    "receive_training_process",
                    json.dumps(
                        {
                            "response": response,
                            "labId": data["labId"],
                            "trainId": data["trainId"],
                        }
                    ),
                )
                await asyncio.sleep(0.1)
        # Use asyncio.wait_for to apply the timeout
        await asyncio.wait_for(func_with_timeout(), timeout=TIMEOUT_LIMIT)
    except asyncio.TimeoutError:
        await sio.emit(
            "receive_training_process",
            json.dumps(
                {
                    "response": {
                        "message": "Timeout. Please try again later.",
                        "status": False
                    },
                    "labId": data["labId"],
                    "trainId": data["trainId"],
                }
            ),
        )
    
async def start_testing(data):
    try:
        async def func_with_timeout():
            response = await test(
                data["test_data_dir"],
                data["labId"],
                data["ckpt_number"],
                data["model_type"],
                data["embed_size"],
                data["num_neurons"],
                data["num_layers"],
                data["backbone"],
            )
            await sio.emit(
                f"receive_testing_process",
                json.dumps(
                    {"response": response, "labId": data["labId"], "testId": data["testId"]}
                ),
            )
            await sio.sleep(0.1)

        # Use asyncio.wait_for to apply the timeout
        await asyncio.wait_for(func_with_timeout(), timeout=TIMEOUT_LIMIT)
    except asyncio.TimeoutError:
        await sio.emit(
            "receive_testing_process",
            json.dumps(
                {
                    "response": {
                        "message": "Timeout. Please try again later.",
                        "status": False
                    },
                    "labId": data["labId"],
                    "testId": data["testId"]
                }
            ),
        )

async def start_infering(data):
    try:
        async def func_with_timeout():
            response = await infer(
                data["url_sample"],
                data["labId"],
                data["ckpt_number"],
                data["model_type"],
                data["embed_size"],
                data["num_neurons"],
                data["num_layers"],
                data["backbone"],
            )
            await sio.emit(
                f"receive_infering_process",
                json.dumps(
                    {"response": response, "labId": data["labId"], "inferId": data["inferId"]}
            ))
            await sio.sleep(0.1)
        await asyncio.wait_for(func_with_timeout(), timeout=TIMEOUT_LIMIT)
    except asyncio.TimeoutError:
        await sio.emit(
            "receive_infering_process",
            json.dumps(
                {
                    "response": {
                        "message": "Timeout. Please try again later.",
                        "status": False
                    },
                    "labId": data["labId"],
                    "inferId": data["inferId"]
                }
            ),
        )

async def transform_compare_phase_testing_thread(data):
    try:
        async def func_with_timeout():
            responseComparing = await test(
                data["test_data_dir"],
                data["labId"],
                None,
                data["model_type"],
                data["embed_size"],
                data["num_neurons"],
                data["num_layers"],
                data["backbone"],
                data["sample_model_dir"],
            )
            await sio.emit(
                f"finish_compare_phase_testing",
                json.dumps(
                    {
                        "responseComparing": responseComparing,
                        "labId": data["labId"],
                        "compareId": data["compareId"],
                    }
                ),
            )
            await sio.sleep(0.1)
        await asyncio.wait_for(func_with_timeout(), timeout=TIMEOUT_LIMIT)
    except asyncio.TimeoutError:
        await sio.emit(
            "finish_compare_phase_testing",
            json.dumps(
                {
                    "response": {
                        "message": "Timeout. Please try again later.",
                        "status": False
                    },
                    "labId": data["labId"],
                    "compareId": data["compareId"],
                }
            ),
        )

async def transform_compare_phase_infering_thread(data):
    try:
        async def func_with_timeout():
            responseComparing = await infer(
                data["url_sample"],
                data["labId"],
                None,
                data["model_type"],
                data["embed_size"],
                data["num_neurons"],
                data["num_layers"],
                data["backbone"],
                data["sample_model_dir"],
            )
            await sio.emit(
                f"finish_compare_phase_infering",
                json.dumps(
                    {
                        "responseComparing": responseComparing,
                        "labId": data["labId"],
                        "compareId": data["compareId"],
                    }
                ),
            )
            await sio.sleep(0.1)
        await asyncio.wait_for(func_with_timeout(), timeout=TIMEOUT_LIMIT)
    except asyncio.TimeoutError:
        await sio.emit(
            "finish_compare_phase_infering",
            json.dumps(
                {
                    "response": {
                        "message": "Timeout. Please try again later.",
                        "status": False
                    },
                    "labId": data["labId"],
                    "compareId": data["compareId"],
                }
            ),
        )

async def start_reviewing_dataset(data):
    numTrain = get_test_review_data(
        data["data_dir"]
    )
    numTest = get_test_review_data(data["test_data_dir"])

    await sio.emit(
        f"receive_reviewing_dataset_process",
        json.dumps(
            {
                "response": {
                    "numTrain": numTrain,
                    "numTest": numTest,
                },
                "datasetId": data["datasetId"],
            }
        ),
    )
    await sio.sleep(0.1)

@sio.on("start_training")
async def start_training_listener(data, background_tasks: BackgroundTasks):
    # Thread(target=await start_training(data)).start()
    background_tasks.add_task(run_async, start_training, data)


@sio.on("start_testing")
async def start_testing_listener(data):
    Thread(target=await start_testing(data)).start()


@sio.on("start_infering")
async def start_infering_listener(data):
    Thread(target=await start_infering(data)).start()



@sio.on("start_reviewing_dataset")
async def start_reviewing_dataset_listener(data):
    Thread(target=await start_reviewing_dataset(data)).start()

@sio.on("transform_compare_phase_testing")
async def transform_compare_phase_testing_listener(data):
    Thread(target=await transform_compare_phase_testing_thread(data)).start()


@sio.on("transform_compare_phase_infering")
async def transform_compare_phase_infering_listener(data):
    Thread(target=await transform_compare_phase_infering_thread(data)).start()


@sio.event
async def disconnect():
    print("disconnected from server")


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        debug=True,
        ws_ping_interval=99999999,
        ws_ping_timeout=99999999,
    )
