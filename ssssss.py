import socketio
import time
import asyncio
import logging
import ssl
import certifi

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create a Socket.IO client
sio = socketio.Client(
    logger=True,
    engineio_logger=True,
    reconnection=True,
    reconnection_attempts=3,
    reconnection_delay=1,
)


@sio.event
def connect():
    logger.info("Connected to server!")
    logger.info(f"Transport method: {sio.transport()}")
    logger.info(f"SID: {sio.get_sid()}")


@sio.event
def connect_error(data):
    logger.error(f"Connection failed: {data}")


@sio.event
def disconnect():
    logger.info("Disconnected from server")


@sio.on("data")
def on_data(data):
    logger.info(f"Received data: {data}")


async def test_server():
    SERVER_URL = "https://e82e-98-97-27-170.ngrok-free.app"

    try:
        logger.info(f"Attempting to connect to server at {SERVER_URL}...")

        # Connect with both WebSocket and polling as fallback
        sio.connect(
            SERVER_URL,
            transports=["websocket"],
        )

        while asyncio.get_event_loop().is_running():
            sio.emit("message", "Hello, world!")
            # Keep the connection alive for a while to receive data
            await asyncio.sleep(10)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(f"Error type: {type(e)}")
    finally:
        if sio.connected:
            sio.disconnect()


if __name__ == "__main__":
    print("Socket.IO Test Client Starting...")
    print("=" * 50)
    asyncio.run(test_server())
