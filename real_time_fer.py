import argparse
import numpy as np
from keras.models import load_model
import cv2


class RealTimeFER():
	def __init__(self, model_path='./models/model.h5'):
		self.emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
		self.model = load_model(model_path)

		cv2.namedWindow("preview")

		self.cap = cv2.VideoCapture(0)

	def detect_facial_emotion(self):
		while(self.cap.isOpened()):
			ret, frame = self.cap.read()

			if frame is not None:
				cv2.imshow("preview", frame)

			if ret:
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

				face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
				faces = face_cascade.detectMultiScale(gray, 1.3, 5)

				for (x, y, w, h) in faces:
					cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
					roi_gray = gray[y:y + h, x:x + w]
					cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
					cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
					prediction = self.model.predict(cropped_img)
					cv2.putText(frame, self.emotion_dict[int(np.argmax(prediction))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

					cv2.imshow('frame', frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		self.cap.release()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--model', help='Model path', required=False)
	args = vars(parser.parse_args())

	if args['model']:
		rtfer = RealTimeFER(args['model'])
	else:
		rtfer = RealTimeFER()
		rtfer.detect_facial_emotion()