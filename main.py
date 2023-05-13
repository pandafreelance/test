from kivy.graphics.texture import Texture
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.toolbar import MDTopAppBar
from kivy.uix.image import Image
from kivy.clock import Clock
import numpy as np
import tensorflow as tf
import pyttsx3 as pt

import cv2

class MainApp(MDApp):

    def build(self):
        layout = MDBoxLayout(orientation="vertical",
                             md_bg_color="#2c3e50"
                             )
        self.face_cascade = cv2.CascadeClassifier('./assets/PanneauDetectionV7.xml')
        # Load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path="./assets/DetectionPanneauxV4.tflite")
        self.interpreter.allocate_tensors()
        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.reader = pt.init()
        self.reader.setProperty('volume', 1.0)
        self.reader.setProperty('rate', 210)
        self.reader.setProperty("voice", "french")

        layout.add_widget(MDTopAppBar(
            title="Panneaux de signalisation",
            md_bg_color="#e67e22"
        ))

        self.interpreter1 = tf.lite.Interpreter(model_path="./assets/ReconnaissancePanneauxSans43V8.tflite")
        self.interpreter1.allocate_tensors()
        # Get input and output tensors.
        self.input_details1 = self.interpreter1.get_input_details()
        self.output_details1 = self.interpreter1.get_output_details()

        self.image = Image()
        layout.add_widget(self.image)

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.load_video, 1.0/30.0)

        return layout

    def load_video(self, *args):
        ret, frame = self.capture.read()
        self.face_det(frame)
        #frame initialize
        self.image_frame = frame
        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture



    def face_det(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        objects = self.face_cascade.detectMultiScale(gray, 1.1, 15)
        for object in objects:
            print("face")
            x, y, w, h = object
            panneau_img = gray[y:y+h, x:x+w]
            panneau_img1 = cv2.resize(panneau_img, (35, 35))
            panneau_img = cv2.resize(panneau_img, (35, 35))
            panneau_img1 = self.interpolationVoision(panneau_img1, 35, 35)
            panneau_img1 = np.array(panneau_img1, dtype=np.float32)
            panneau_img1 = np.expand_dims(panneau_img1, axis=0)
            panneau_img = self.interpolationVoision(panneau_img, 35, 35)
            panneau_img = np.expand_dims(panneau_img, axis=0)
            panneau_img = np.array(panneau_img, dtype=np.float32)
            # result2 = np.argmax(model1.predict(panneau_img2))
            # input_details[0]['index'] = the index which accepts the input
            self.interpreter.set_tensor(self.input_details[0]['index'], panneau_img1)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            choix = output_data
            if choix >0.8:
                self.interpreter1.set_tensor(self.input_details1[0]['index'], panneau_img)
                self.interpreter1.invoke()
                output_data1 = self.interpreter1.get_tensor(self.output_details1[0]['index'])
                choix1 = np.argmax(output_data1)
                self.parle(choix1)
                cv2.rectangle(frame, (int(x), int(y)), (int((x + w)), int((y + h))), (0, 255, 0), 2)
        #cv2.imshow("Image", frame)

    def interpolationVoision(self, A, x, y):
        n = A.shape
        if len(n) >= 3:
            B = np.zeros((x, y, 3))
            if n[2] == 4:  # changement de dimension de 4 a 3
                A = A[:, :, :3]
        else:
            B = np.zeros((x, y, 3))
        posx = np.arange(x) * (n[0] / x)
        posy = np.arange(y) * (n[1] / y)
        for i in range(len(posx)):
            for j in range(len(posy)):
                # B[i,j] = A[int(i*posx - 1 ),int(j*posy - 1)]
                B[i, j] = A[int(posx[i]), int(posy[j])]
        return B

    def parle(self, texte):
        self.reader.say(texte)
        self.reader.runAndWait()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    MainApp().run()

