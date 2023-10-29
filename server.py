import queue
import cv2
import numpy as np
import os
from flask import Flask, Response, jsonify
import time
import threading


app = Flask(__name__)


cap = cv2.VideoCapture(0)
#cap  = cv2.VideoCapture('sample_vids/chair_right.mp4')
frame_queue = queue.Queue(maxsize=10)

# Load MobileNet SSD model
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "laptop"]


latest_direction = "No guidance available"
latest_frame = None  
frame_lock = threading.Lock()

def continuous_detection():
    global latest_direction, latest_frame
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        start_time = time.time()  # Start time for FPS calculation
        direction, processed_frame = detect_and_navigate(frame)
        end_time = time.time()  # End time for FPS calculation

        fps = 1 / (end_time - start_time)  # Calculate FPS
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        with frame_lock:
            latest_direction = direction
            latest_frame = processed_frame


# Start the continuous detection thread
# detection_thread = threading.Thread(target=continuous_detection)
# detection_thread.start()


def detect_and_navigate(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] in ["chair", "laptop","tvmonitor"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Determine direction
                center_x = (startX + endX) // 2
                if center_x < w // 3:
                    direction = "Turn slightly left"
                elif center_x > 2 * w // 3:
                    direction = "Turn slightly right"
                else:
                    direction = "Go straight"
                
                print(direction)
                return direction, frame

    return "No guidance available", frame

def display_output():
    while True:
        ret, frame = cap.read()
        
        # Check if the video has ended
        # if not ret:
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the beginning
        #     continue  # Skip the rest of the loop and start from the beginning

        # frame_queue.put(frame)
        if not frame_queue.full():
            frame_queue.put(frame)

        if latest_frame is not None:
            cv2.imshow("Output", latest_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        time.sleep(0.06)
    cap.release()            
    cv2.destroyAllWindows()

# @app.route('/video_feed')
# def video_feed():
#     def generate():
#         while True:
#             ret, frame = cap.read()
#             if ret:
#                 _, jpeg = cv2.imencode('.jpg', frame)
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
#     return Response(generate(), content_type='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    def generate():
        global latest_frame
        while True:
            if frame_lock.acquire(False):  # Try to acquire the lock, but don't block
                if latest_frame is not None:
                    #_, jpeg = cv2.imencode('.jpg', latest_frame)


                    compressed_frame = cv2.resize(latest_frame, (640, 480))  # Change to desired resolution
                    
                    # Compress the frame with reduced JPEG quality
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 45]  # Quality range is 0-100; lower means more compression
                    _, jpeg = cv2.imencode('.jpg', compressed_frame, encode_param)
                    
                    frame_lock.release() 
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                else:
                    frame_lock.release()
            else:
                time.sleep(0.08)  # If we couldn't get the lock, wait a bit before trying again
    return Response(generate(), content_type='multipart/x-mixed-replace; boundary=frame')



@app.route('/direction')
def get_direction():
    with frame_lock:
        return jsonify({"direction": latest_direction})

def start_flask_app():
    # Start the Flask app
    app.run(host='0.0.0.0', port=5005, threaded=True)

# if __name__ == '__main__':
#     # Start the continuous detection thread
#     detection_thread = threading.Thread(target=continuous_detection)
#     detection_thread.start()

#     # Call the display function in the main thread
#     display_output()

#     # Start the Flask app
#     app.run(host='0.0.0.0', port=5000, threaded=True)

if __name__ == '__main__':
    # Start the continuous detection thread
    detection_thread = threading.Thread(target=continuous_detection)
    detection_thread.start()

    # Start the Flask app in a separate thread
    flask_thread = threading.Thread(target=start_flask_app)
    flask_thread.start()

    # Call the display function in the main thread
    display_output()