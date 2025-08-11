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
def infer_d_from_yh(y0, h0, h_values, d_values, y_matrix, spline_order=2):
    h = np.asarray(h_values)
    d = np.asarray(d_values)
    y = np.asarray(y_matrix)
    spline = RectBivariateSpline(h, d, y, kx=spline_order, ky=spline_order) #Convert to spline functions

    def objective(d_val):
        return spline(h0, d_val)[0][0] - y0 # Define function: F(h0,d0) = y0 

    d_min, d_max = np.min(d), np.max(d)
    f_min, f_max = objective(d_min), objective(d_max) # Define d range

    if f_min * f_max > 0:
        return float('nan')  # Return NaN if y input is out of range

    try:
        sol = root_scalar(objective, bracket=[d_min, d_max], method='brentq') # Find d using Brent's method
        return float(sol.root)  # Output distance !!!
    except Exception:
        return float('nan')  # Return NaN if root finding fails


# Distance setup for interpolation
h_values = np.array([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
d_values = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
y_matrix = np.array([[0.76497, 0.67318, 0.62728, 0.59896, 0.5804 , 0.56771, 0.55859, 0.55143,
        0.5459 , 0.54102, 0.53646, 0.53353, 0.5306 , 0.52799, 0.52539, 0.52311],
       [0.79199, 0.69271, 0.64193, 0.6123 , 0.59277, 0.57878, 0.56868, 0.5612 ,
        0.55469, 0.54948, 0.54492, 0.54134, 0.53841, 0.53548, 0.53288, 0.5306 ],
       [0.81673, 0.70996, 0.65495, 0.6237 , 0.60221, 0.58724, 0.5765 , 0.56836,
        0.56152, 0.55566, 0.55111, 0.54688, 0.54362, 0.54069, 0.53809, 0.53548],
       [0.83464, 0.72201, 0.66374, 0.62923, 0.60677, 0.59049, 0.5791 , 0.57031,
        0.56315, 0.55697, 0.55176, 0.54395, 0.54069, 0.53776, 0.53776, 0.53483],
       [0.85547, 0.73438, 0.67285, 0.63639, 0.61263, 0.59538, 0.58366, 0.57389,
        0.56608, 0.55957, 0.55404, 0.54948, 0.5459 , 0.54232, 0.53939, 0.53646],
       [0.87858, 0.74902, 0.68392, 0.64486, 0.61882, 0.60091, 0.58757, 0.57747,
        0.56934, 0.56217, 0.55632, 0.55176, 0.54753, 0.54395, 0.54069, 0.53776],
       [0.90169, 0.76432, 0.69434, 0.65299, 0.62598, 0.60645, 0.59277, 0.58171,
        0.57324, 0.56641, 0.55957, 0.55436, 0.5498 , 0.54655, 0.54264, 0.53939]])


# Open video capture
cap = cv2.VideoCapture(2) 
# Insert camera height
camera_height = 0.9
# Insert focal length in mm
focal_length = 23
# Initialize track history for speed calculation
track_history = {}
# Initialize clock
t0 = time.time()
# Initialize alert status
alert_working = False

if not cap.isOpened():
    print("Error: Could not open video source")
    exit()

# Set up UDP for reveiving pitch angle
UDP_IP = "xx.xx.xx.xx" # Insert UDP IP address
UDP_PORT = xxxx # Insert UDP port selected 
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(True)

# # Receive pitch angle via UDP
def udp_listener():
    global pitch_angle
    while True:
        data, _ = sock.recvfrom(1024)
        try:
            pitch_angle = float(data.decode())
        except ValueError:
            continue
# Start UDP listener in a separate thread to reduce delay
listener = threading.Thread(target=udp_listener, daemon=True)
listener.start()

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
            img_with_boxes = result.orig_img.copy()
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
                D_y = infer_d_from_yh(y_nor, camera_height, h_values, d_values, y_matrix)

                # Calculate Dx
                D_h = (D_y ** 2 + camera_height ** 2) ** 0.5
                D_x = (x_nor- 0.5)  / (0.7396 * D_h ** -1.131) 

                # Calculate real distance
                real_dis = (D_y ** 2 + D_x ** 2) ** 0.5
                
                # Calculate relative velocity
                speed = None
                v_rel = None 
                track_id = int(box.id.item()) if box.id is not None else None
                now = time.time()

                if track_id is not None and D_y is not None and D_x is not None:
                    X, Y = D_x, D_y  # Current actual position
                    if track_id in track_history:
                        t_prev, X_prev, Y_prev = track_history[track_id]
                        dt = now - t_prev # Time difference since last position
                        # Calculate speed if time difference is valid
                        if dt > 0.05:  
                            dX = X - X_prev
                            dY = Y - Y_prev
                            # Calculate pedestrian d_a
                            d_a = np.array([dX, dY])
                            d_r_unit = np.array([X_prev, Y_prev]) / -np.linalg.norm([X_prev, Y_prev])
                            v_rel = np.dot(d_a, d_r_unit)  / dt  # Relative velocity
                            hit_t = real_dis / v_rel # Predicted time to hit
                            speed = v_rel

                    # Update track history
                    track_history[track_id] = (now, X, Y)

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
                    if hit_t > 0 and hit_t < 2 and v_rel < 5: #triger alert
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
                cv2.rectangle(img_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Draw bounding box label
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                cv2.rectangle(
                    img_with_boxes,
                    (int(x1), int(y1) - text_size[1] - 5),
                    (int(x1) + text_size[0], int(y1) - 5),
                    color, -1
                )
                cv2.putText(
                    img_with_boxes, label,
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

            x_pos = img_with_boxes.shape[1] - text_size[0] - 10
            y_pos = img_with_boxes.shape[0] - 10

            cv2.rectangle(img_with_boxes,
                          (x_pos - 5, y_pos - text_size[1] - 5),
                          (x_pos + text_size[0] + 5, y_pos + 5),
                          (255, 255, 255), -1)

            cv2.putText(img_with_boxes, alert_text, (x_pos, y_pos),
                        font, font_scale, color, thickness)

        # Display angle text
        if abs(pitch_angle) > 100 or abs(pitch_angle) < 80:
            angle_text = f"Angle= {pitch_angle}, out of range"
            angle_in_range = False
        else:
            angle_text = f"Angle= {pitch_angle}"
            angle_in_range = True
        h = f"h= {camera_height:.2f}m"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (127, 255, 0)
        x_pos = 5
        y_pos = 25

        cv2.putText(img_with_boxes, angle_text, (x_pos, y_pos), font, font_scale, color, thickness)
        cv2.putText(img_with_boxes, h, (x_pos, y_pos + 30), font, font_scale, color, thickness)

        # Calculate elapsed time and display it
        elapsed = int(time.time() - t0)     
        mins, secs = divmod(elapsed, 60)  
        start_time = f"{mins:02d}:{secs:02d}"

        time_shown = f"{start_time}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(time_shown, font, font_scale, thickness)
        x_pos = img_with_boxes.shape[1] - text_w - 10 
        y_pos = text_h + 10

        cv2.putText(
            img_with_boxes,
            time_shown,
            (x_pos, y_pos),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )
        # Show the processed frame in a window
        cv2.imshow('Object Detection', img_with_boxes)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    except Exception as e:
        print("Error in main loop:", e)

cap.release()
cv2.destroyAllWindows()

