import socketio
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter, FastAPI
import random
import logging
import sys

logger = logging.getLogger(__name__)
# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

# Set specific loggers to DEBUG level
logging.getLogger("engineio").setLevel(logging.DEBUG)
logging.getLogger("engineio.server").setLevel(logging.DEBUG)
logging.getLogger("socketio").setLevel(logging.DEBUG)
logging.getLogger("socketio.server").setLevel(logging.DEBUG)
logging.getLogger("websockets").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

buffer = []


def build_app() -> socketio.ASGIApp:
    #           Managers           #
    #                              #
    ################################
    router = APIRouter(prefix="/api/v1")

    app = FastAPI(cors_allow_origins=["*"])

    app.include_router(router)
    origins = [
        "*",
        "https://e82e-98-97-27-170.ngrok-free.app",
        "wss://e82e-98-97-27-170.ngrok-free.app",
        "http://localhost:8000",
        "ws://localhost:8000",
        "http://localhost:3000",
        "https://drone-detection-viewer.vercel.app/",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    sio = socketio.AsyncServer(
        cors_allowed_origins=origins,
        # https://github.com/pyropy/fastapi-socketio/issues/13 but this
        # also references 205 in python-socketio repo below
        # because we're using CORSMiddleware in the fastapi server
        async_mode="asgi",
        logger=True,
        engineio_logger=True,
        # mount_location="/socket.io",
        transports=["websocket", "polling"],
        always_connect=True,
    )

    @sio.event
    async def connect(sid, environ):
        logger.info(f"Client {sid} connected")

    @sio.event
    async def disconnect(sid):
        logger.info(f"Client {sid} disconnected")

    async def stream_output():
        while True:
            # while len(buffer) > 0:
            #     data = buffer.pop(0)
            #     await sio.emit("data", data)
            await sio.sleep(1)
            for i in range(3):
                data = {"probability": random.randint(0, 100), "id": i}
                await sio.emit("data", data)  # Match the event name used in the client

    @sio.on("message")
    async def message(sid, data):
        logger.info(f"Got message from client {sid}")
        logger.info(f"Message data: {data}")
        buffer.append(data)

    @sio.on("start_stream")
    async def start_stream(sid):
        logger.info(f"Starting stream for client {sid}")
        asyncio.create_task(stream_output())  # Start streaming data

    return socketio.ASGIApp(sio, app)


app = build_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
