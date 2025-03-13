import tkinter as tk
from tkinter import messagebox
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import threading
from PIL import Image, ImageTk

class DrowsinessDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Drowsiness Driver Detection")
        
        # Set screen width and height
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Load the image using PIL
        pil_image = Image.open("background_image.png")
        
        # Resize the image to half of the screen width and full height
        pil_image = pil_image.resize((screen_width // 2, screen_height), resample=Image.LANCZOS)
        
        # Convert PIL image to tkinter PhotoImage
        self.background_image = ImageTk.PhotoImage(pil_image)
        
        # Create a canvas and display the background image on the left half of the screen
        self.canvas = tk.Canvas(root, width=screen_width // 2, height=screen_height)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.background_image)
        
        # Create a frame on the right half of the screen for buttons
        self.button_frame = tk.Frame(root, width=screen_width // 2, height=screen_height, bg='lightgrey')
        self.button_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        
        # Add a label for heading
        self.label = tk.Label(self.button_frame, text="Drowsiness Driver Detection", font=('Helvetica', 16, 'bold'), bg='lightgrey')
        self.label.pack(pady=(50, 20))
        
        # Add a start button
        self.start_button = tk.Button(self.button_frame, text="Start Detection", command=self.start_detection, bg='green', fg='white', font=('Helvetica', 14, 'bold'))
        self.start_button.pack(pady=20)
        
        # Add a quit button
        self.quit_button = tk.Button(self.button_frame, text="Quit", command=self.quit_app, bg='red', fg='white', font=('Helvetica', 14, 'bold'))
        self.quit_button.pack(pady=20)
        
        # Initialize mixer for sound
        self.mix = mixer.init()
        self.mix = mixer.music.load("music.wav")
        
        # Set parameters for drowsiness detection
        self.thresh = 0.25
        self.frame_check = 20
        self.detect = dlib.get_frontal_face_detector()
        self.predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        self.flag = 0
        self.cap = cv2.VideoCapture(0)
        
        # Check if camera is opened successfully
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open camera. Please check if the camera is connected and permissions are granted.")
            return
        
        # Set font and color for text overlay
        self.text_font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_color = (0, 255, 0)
        self.text_thickness = 2
        self.text_pos_awake = (20, 40)
        self.text_pos_drowsy = (20, 80)
        self.text_awake = "Driver is Awake"
        self.text_drowsy = "Drowsiness Detected!"

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def play_sound_and_display_message(self):
        self.mix = mixer.music.play()
        messagebox.showwarning("Drowsiness Detected", "Drowsiness Detected!\nPlease take a break or stop driving.")

    def detect_drowsiness(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to retrieve frame from camera.")
                break

            frame = imutils.resize(frame, width=root.winfo_screenwidth() // 2)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            subjects = self.detect(gray, 0)
            for subject in subjects:
                shape = self.predict(gray, subject)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                if ear < self.thresh:
                    self.flag += 1
                    if self.flag >= self.frame_check:
                        self.play_sound_and_display_message()
                        cv2.putText(frame, self.text_drowsy, self.text_pos_drowsy, self.text_font, 1, (0, 0, 255), self.text_thickness, cv2.LINE_AA)
                else:
                    self.flag = 0
                    cv2.putText(frame, self.text_awake, self.text_pos_awake, self.text_font, 1, self.text_color, self.text_thickness, cv2.LINE_AA)
            cv2.imshow("Drowsiness Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        self.cap.release()

    def start_detection(self):
        self.detector_thread = threading.Thread(target=self.detect_drowsiness)
        self.detector_thread.start()

    def quit_app(self):
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DrowsinessDetector(root)
    root.mainloop()
