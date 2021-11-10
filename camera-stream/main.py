import asyncio
import websockets
from camera import DepthCamera
from networks.yolo.yolov3 import YoloNetwork
import io


async def frames(websocket, path):
    while True:
        img, depth = camera.get_frame()
        if img is not None:
            result = nn.predict(img, depth)
            buf = io.BytesIO()
            result.save(buf, format="jpeg")
            await websocket.send(buf.getvalue())
        else:
            print("Skipping frame")


if __name__ == "__main__":
    nn = YoloNetwork()

    camera = DepthCamera()
    camera.start()

    start_server = websockets.serve(frames, "", 5678)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
