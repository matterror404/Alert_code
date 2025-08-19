from ultralytics import YOLO
import cv2
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import root_scalar
import socket
import threading
import time
import math


# Initialize YOLO11n model
model = YOLO('Yolo/yolo11n.pt')

# Interpolation function
def find_d(y_in, h_in, h_arr, d_arr, y_arr):
    h = np.asarray(h_arr)
    d = np.asarray(d_arr)
    y = np.asarray(y_arr)
    spline = RectBivariateSpline(h, d, y, kx=2, ky=2) #Convert to spline functions

    def equation(d_val):
        return float(spline(h_in, d_val) - y_in) # Define function: F(h0,d0) = y_in 
    
    d = root_scalar(equation, bracket=[2, 17], method='brentq') # Find d using Brent's method
    return float(d.root)  # Output distance 


# Distance setup for interpolation
h_cali = np.array([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
d_cali = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
y_cali = np.array([
    [0.764973958, 0.673177083, 0.627278646, 0.598958333, 0.580403646, 0.567708333, 0.55859375, 0.551432292, 0.545898438, 0.541015625, 0.536458333, 0.533528646, 0.530598958, 0.527994792, 0.525390625, 0.523111979],
    [0.791992188, 0.692708333, 0.641927083, 0.612304688, 0.592773438, 0.578776042, 0.568684896, 0.561197917, 0.5546875, 0.549479167, 0.544921875, 0.541341146, 0.538411458, 0.535481771, 0.532877604, 0.530598958],
    [0.816731771, 0.709960938, 0.654947917, 0.623697917, 0.602213542, 0.587239583, 0.576497396, 0.568359375, 0.561523438, 0.555664063, 0.551106771, 0.546875, 0.543619792, 0.540690104, 0.538085938, 0.535481771],
    [0.834635417, 0.722005208, 0.663736979, 0.629231771, 0.606770833, 0.590494792, 0.579101563, 0.5703125, 0.563151042, 0.556966146, 0.551757813, 0.543945313, 0.540690104, 0.537760417, 0.537760417, 0.534830729],
    [0.85546875, 0.734375, 0.672851563, 0.636393229, 0.612630208, 0.595377604, 0.583658854, 0.573893229, 0.566080729, 0.559570313, 0.554036458, 0.549479167, 0.545898438, 0.542317708, 0.539388021, 0.536458333],
    [0.878580729, 0.749023438, 0.683919271, 0.644856771, 0.618815104, 0.600911458, 0.587565104, 0.577473958, 0.569335938, 0.562174479, 0.556315104, 0.551757813, 0.547526042, 0.543945313, 0.540690104, 0.537760417],
    [0.901692708, 0.764322917, 0.694335938, 0.652994792, 0.625976563, 0.606445313, 0.592773438, 0.581705729, 0.573242188, 0.56640625, 0.559570313, 0.554361979, 0.549804688, 0.546549479, 0.542643229, 0.539388021]
])


# Open video capture
cap = cv2.VideoCapture(2) 
# Insert camera height
camera_height = 0.65
# Insert focal length in mm
focal_length = 23
# Initialize track history for speed calculation
track_history = {}
speed_history = {} 
history_length = 5
# Initialize clock
t0 = time.time()
# Initialize alert status
alert_working = False
# Set alert trigger time in seconds
alert_trigger_time = 1

if not cap.isOpened():
    print("Error: Could not open video source")
    exit()
# Using the code in https://github.com/umer0586/SensaGram 
import json
import socket
import threading
import math

# Global variable to store the latest quaternion data
latest_quaternion_data = None
server_running = False
class UDPServer:
    def __init__(self, address, buffer_size=1024):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.address = address
        self.buffer_size = buffer_size

    def setDataCallBack(self, callback):
        self.callback = callback

    def start(self, daemon=False):
        self.thread = threading.Thread(target=self.__listen__)
        self.thread.daemon = daemon  # Daemon thread, so it terminates with the main program
        self.thread.start()

    def stop(self):
        self.sock.close()  # Close the socket to unblock recvfrom

    def __listen__(self):
        self.sock.bind(self.address)
        
        while True:
            try:                    
                data, address = self.sock.recvfrom(self.buffer_size)
                
                if self.callback != None:
                    self.callback(data)

            except socket.error:
                break  # Break if the socket is closed
                 
# Function to update the cube's rotation in the main thread using quaternion
def update_rotation():
    global latest_quaternion_data
    if latest_quaternion_data:
        x, y, z, w = latest_quaternion_data
        global phone
        # Apply quaternion rotation to the cube
        phone.rotation_quaternion = (w, x, y, z)

    # Return a small time interval to keep calling this function
    return 0.01

def pitch_cal(x, y, z, w): # Found in https://github.com/taylorcoders/Quaternion2EulerAngle/blob/master/Quaternion2EulerAngle.py
    pitch_x = 2.0 * (w*y - z*x)
    pitch_z = 1.0 - 2.0 * (y*y + x*x)
    angle = math.atan2(pitch_x, pitch_z)
    return math.degrees(angle)

# The onData callback from the UDP server
def onData(data):
    global latest_quaternion_data, pitch
    jsonData = json.loads(data)
    sensorType = jsonData["type"]
    values = jsonData["values"]
    
    if sensorType == "android.sensor.rotation_vector":
        # Extract the quaternion data (x, y, z, w)
        if len(values) == 4:
            x, y, z, w = values
        else:
            x, y, z = values[:3]
            w = (1.0 - (x*x + y*y + z*z))**0.5  # Estimate w if it's not provided
        # Update the global variable with the latest quaternion values
        latest_quaternion_data = (x, y, z, w)
        # print(f"Received quaternion data: {latest_quaternion_data}")
        pitch = pitch_cal(latest_quaternion_data[0], latest_quaternion_data[1], 
                                      latest_quaternion_data[2], latest_quaternion_data[3])

# Set up UDP for reveiving pitch angle
server = UDPServer(address=("192.168.137.1", 9876))# Insert UDP IP address. Insert UDP port selected 
server.setDataCallBack(onData)
server.start()
server_running = True

# Main
# Process video frames
while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        ysize, xsize = frame.shape[:2]
        results = model.track(frame, persist=True)

        
        for result in results:
            pitch_angle = abs(pitch) # Safe the pitch angle of the next frame
            out_frame = result.orig_img.copy()
            for box in result.boxes:
                # Extract bounding box coordinates and class information
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf.item()
                cls_id = int(box.cls.item())
                class_name = model.names[cls_id]
                # Only process 'person' class
                if class_name != 'person':
                    continue
            # Calculate distance and speed with tbe equations in thesis and interpolation
                # Normalize coordinates
                y_nor = y2 / ysize
                x_nor = (x1 + x2) / (2 * xsize)
                # Convert focallength of other camera to 23 mm (for 4:3 aspect ratio only)
                y_nor = (y_nor - 0.5) * (focal_length / 23) + 0.5
                x_nor = (x_nor - 0.5) * (focal_length / 23) + 0.5
                if x_nor < 0.2 or x_nor > 0.8:
                    continue # Ignore objects if too far left or right, prevent the false x_nor value
                # Angle calibration
                if pitch_angle > 80 and pitch_angle <= 90:
                    dy = (0.0225 * y_nor * y_nor - 0.0258 * y_nor + 0.0229) * (pitch_angle - 90)
                elif pitch_angle > 90 and pitch_angle <= 100:
                    dy = (0.0223 * y_nor * y_nor - 0.0182 * y_nor + 0.02) * (pitch_angle - 90)
                # elif angle out of range
                else: 
                    dy = 0.0
                y_nor = y_nor - dy

                # Calculate Dy using interpolation
                D_y = find_d(y_nor, camera_height, h_cali, d_cali, y_cali)

                # Calculate Dx
                D_h = (D_y ** 2 + camera_height ** 2) ** 0.5
                D_x = (x_nor- 0.5)  / (0.7396 * D_h ** -1.131) 

                # Calculate real distance
                real_dis = (D_y ** 2 + D_x ** 2) ** 0.5
                
                # Calculate relative velocity
                speed = None
                v_rel = None 
                ped_id = int(box.id.item()) if box.id is not None else None
                now = time.time()

                if ped_id is not None and D_y is not None and D_x is not None:
                    X, Y = D_x, D_y  # Current actual position
                    if ped_id in track_history:
                        t_prev, X_prev, Y_prev = track_history[ped_id]
                        dt = now - t_prev # Time difference since last position
                        # Calculate speed if time difference is valid
                        if dt > 0.05:  
                            dX = X - X_prev
                            dY = Y - Y_prev
                            # Calculate pedestrian d_a
                            d_a = np.array([dX, dY])
                            d_r_unit = np.array([X_prev, Y_prev]) / -np.linalg.norm([X_prev, Y_prev])
                            v_rel = np.dot(d_a, d_r_unit)  / dt  # Relative velocity
                            
                            if ped_id not in speed_history:
                                speed_history[ped_id] = []
                            speed_history[ped_id].append(v_rel)
                            
                            if len(speed_history[ped_id]) > history_length:
                                speed_history[ped_id].pop(0) 
                            speed = sum(speed_history[ped_id]) / len(speed_history[ped_id]) # Smooth speed by averaging 5 frame 
                            hit_t = real_dis / speed # Predicted time to hit
                    
                    # Update track history
                    track_history[ped_id] = (now, X, Y)

                class_initial = class_name[0].lower()
                label = class_initial

                # Determine color based on distance and speed
                if D_y > 17:
                    color = (255, 255, 255)
                    label = class_initial
                    alert_working = False
                
                if math.isnan(D_y):
                    color = (255, 255, 255)
                    label = class_initial
                    alert_working = False

                if D_y <= 17:
                    if hit_t > 0 and hit_t < alert_trigger_time and D_y > 2.5 and D_y <= 12: #triger alert
                        color = (0, 0, 255)
                        alert_working = True
                    else:
                        alert_working = False
                        if angle_in_range is False:
                            color = (0, 0, 255)
                        else:
                            color = (0, 255, 0)

                    # Display distance and speed if available
                    if angle_in_range is True:
                        if not math.isnan(D_y):
                            label = f"{real_dis:.1f}m,{speed:.1f}m/s"
                        else:
                            label = f"{class_initial}"

                # Create bounding box
                cv2.rectangle(out_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Draw bounding box label
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                cv2.rectangle(
                    out_frame,
                    (int(x1), int(y1) - text_size[1] - 5),
                    (int(x1) + text_size[0], int(y1) - 5),
                    color, -1
                )
                cv2.putText(
                    out_frame, label,
                    (int(x1), int(y1) - 5),
                    font, font_scale, (0, 0, 0), thickness
                )

        # alert
        if alert_working:
            alert_text = "Warning"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            color = (0, 0, 255)
            text_size = cv2.getTextSize(alert_text, font, font_scale, thickness)[0]

            x_pos = out_frame.shape[1] - text_size[0] - 10
            y_pos = out_frame.shape[0] - 10

            cv2.rectangle(out_frame,
                          (x_pos - 5, y_pos - text_size[1] - 5),
                          (x_pos + text_size[0] + 5, y_pos + 5),
                          (255, 255, 255), -1)

            cv2.putText(out_frame, alert_text, (x_pos, y_pos),
                        font, font_scale, color, thickness)

        # Display angle text
        if abs(pitch_angle) > 100 or abs(pitch_angle) < 80:
            angle_text = f"Angle= {pitch_angle:.1f}, out of range"
            angle_in_range = False
        else:
            angle_text = f"Angle= {pitch_angle:.1f}"
            angle_in_range = True
        h = f"h={camera_height:.2f}m"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (127, 255, 0)
        x_pos = 5
        y_pos = 25

        cv2.putText(out_frame, angle_text, (x_pos, y_pos), font, font_scale, color, thickness)
        cv2.putText(out_frame, h, (x_pos, y_pos + 30), font, font_scale, color, thickness)

        # Show the processed frame in a window
        cv2.imshow('Object Detection', out_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    except Exception as e:
        print("Error in main loop:", e)

cap.release()
cv2.destroyAllWindows()
