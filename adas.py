import cv2
import numpy as np
from ultralytics import YOLO

#its because we only focus on roads
def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]

    #define the triangle
    polygon = np.array([[
        (0,height), #bottom left corner
        (width,height),#bottom right corner
        (width//2,int(height*0.75)) #top centre   
    ]])

    #create a black mask
    mask = np.zeros_like(img)

    #fill the polygon with white color
    cv2.fillPoly(mask, polygon, 255)

    # Apply mask
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image

cap = cv2.VideoCapture("demo.mp4")  # put your video path if its not running otherwise put in same folder
model = YOLO("yolov8n.pt")


while True:
    ret, frame = cap.read()

    if not ret:
        print("Video ended or error")
        break

# Grayscale(Convert the real image to image with edges only)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #removing blur(smoothing the video to make the video less shaky)
    blur = cv2.GaussianBlur(gray,(5,5), 0)
    
    #Edges(highlights the edges whether strong or weak first filter for weak and second for strong)
    edges = cv2.Canny(blur, 30, 100)
    cropped_edges = region_of_interest(edges)
    
    #lane detection logic
    lines = cv2.HoughLinesP(cropped_edges,
                            1,
                            np.pi/180,
                            threshold=30,
                            minLineLength=50,
                            maxLineGap=150)
    
    #slope filtering + averaging + drawing here
    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
             
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            
            if abs(slope) < 0.5 or abs(slope) > 2:
                continue

            if slope < 0:
             left_lines.append((x1,y1,x2,y2))
            else:
                right_lines.append((x1, y1, x2, y2))

    def average_lines(lines):
        if len(lines) == 0:
            return None
        
        x_coords = []
        y_coords = []

        for x1, y1, x2, y2 in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])

        slope, intercept = np.polyfit(x_coords, y_coords, 1)

        return slope, intercept

    def make_line(slope, intercept, height):
        y1 = height
        y2 = int(height * 0.75)

        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        return x1, y1, x2, y2

    left_fit = average_lines(left_lines)
    right_fit = average_lines(right_lines)

    if left_fit is not None:
        x1, y1, x2, y2 = make_line(left_fit[0], left_fit[1], frame.shape[0])
        cv2.line(frame, (x1,y1), (x2,y2), (255,0,0), 5)

    if right_fit is not None:
        x1, y1, x2, y2 = make_line(right_fit[0], right_fit[1], frame.shape[0])
        cv2.line(frame, (x1,y1), (x2,y2), (0,0,255), 5)

    # ---- YOLO DETECTION ----
    results = model(frame)
    frame = results[0].plot()

    #collision detection
    frame_center = frame.shape[1] // 2

    for box in results[0].boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        box_area = (x2 - x1) * (y2 - y1)
        box_center = (x1 + x2) // 2
        box_height = y2 - y1

        if cls in [2, 3,5]:  #cars.bikes,bus
            if (box_area > 30000 and 
                box_height > frame.shape[0] * 0.2 and
                abs(box_center - frame_center) < 150):
                cv2.putText(frame, "COLLISION WARNING!", (frame.shape[1] // 2 - 180, 50),
                             cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0,0,255), 3)

    cv2.imshow("ADAS", frame)

    if cv2.waitKey(15) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()