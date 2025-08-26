

# import cv2
# import numpy as np
# import pandas as pd
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from skimage.feature import hog
# from tkinter import Tk, Label, Button, filedialog, Canvas
# from PIL import Image, ImageTk
# import os

# class RiceGrainClassifier:
#     def __init__(self):
#         self.model = None
#         self.scaler = None
#         self.rice_types = ['Aromatic Heirloom Varieties', 'Deepwater Rice', 'Neang Minh', 'Phka Romduol', 'Phka Rumdeng','Jasmine Rice','Glutinous','White Rice','Red Rice','Black Rice','Organic','Flooded Rice']
#         self.root = Tk()
#         self.root.title("Rice Grain Classifier")
#         self.canvas = None
#         self.image_label = None
#         self.setup_gui()

#     def extract_features(self, grain_image):
#         # Resize grain image to fixed size
#         grain_image = cv2.resize(grain_image, (64, 64))

#         # Convert to grayscale
#         gray = cv2.cvtColor(grain_image, cv2.COLOR_BGR2GRAY)

#         # Extract HOG features
#         hog_features = hog(gray, orientations=8, pixels_per_cell=(16, 16),
#                            cells_per_block=(1, 1), visualize=False, feature_vector=True)

#         # Extract shape features
#         contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if contours:
#             contour = max(contours, key=cv2.contourArea)
#             area = cv2.contourArea(contour)
#             perimeter = cv2.arcLength(contour, True)
#             circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
#         else:
#             area, perimeter, circularity = 0, 0, 0

#         # Extract color features (mean RGB)
#         color_mean = np.mean(grain_image, axis=(0, 1))

#         features = np.concatenate([hog_features, [area, perimeter, circularity], color_mean])

#         # Assert feature length = 134 (128 + 3 + 3)
#         assert len(features) == 134, f"Feature length is {len(features)} instead of 134"

#         return features

#     def preprocess_image(self, image_path):
#         image = cv2.imread(image_path)
#         if image is None:
#             return None, None

#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         denoised = cv2.fastNlMeansDenoising(gray)
#         thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                        cv2.THRESH_BINARY_INV, 11, 2)
#         return image, thresh

#     def detect_grains(self, image, thresh):
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         grains = []
#         for contour in contours:
#             if cv2.contourArea(contour) > 50:
#                 x, y, w, h = cv2.boundingRect(contour)
#                 grain = image[y:y+h, x:x+w]
#                 grains.append((grain, (x, y, w, h)))
#         return grains

#     def train_model(self):
#         # Simulate training data with correct feature length 134
#         X, y = [], []
#         for i, rice_type in enumerate(self.rice_types):
#             for _ in range(20):
#                 features = np.random.rand(134)  # Match extract_features length
#                 X.append(features)
#                 y.append(i)

#         self.scaler = StandardScaler()
#         X_scaled = self.scaler.fit_transform(X)

#         self.model = SVC(kernel='rbf', probability=True)
#         self.model.fit(X_scaled, y)

#     def classify_grains(self, grains):
#         if self.model is None:
#             self.train_model()

#         results = []
#         for grain, (x, y, w, h) in grains:
#             features = self.extract_features(grain)
#             features_scaled = self.scaler.transform([features])
#             pred = self.model.predict(features_scaled)[0]
#             confidence = np.max(self.model.predict_proba(features_scaled)[0])
#             results.append((self.rice_types[pred], confidence, (x, y, w, h)))

#         return results

#     def save_results(self, results, output_path):
#         df = pd.DataFrame([(r[0], r[1], r[2][0], r[2][1], r[2][2], r[2][3])
#                            for r in results],
#                           columns=['Rice_Type', 'Confidence', 'X', 'Y', 'Width', 'Height'])
#         df.to_csv(output_path, index=False)

#         counts = df['Rice_Type'].value_counts()
#         with open(os.path.splitext(output_path)[0] + '_summary.txt', 'w') as f:
#             f.write("Rice Type Counts:\n")
#             for rice_type, count in counts.items():
#                 f.write(f"{rice_type}: {count}\n")
#     def process_image(self):
#         file_path = filedialog.askopenfilename()
#         if not file_path:
#             return

#         image, thresh = self.preprocess_image(file_path)
#         if image is None:
#             return

#         grains = self.detect_grains(image, thresh)
#         results = self.classify_grains(grains)

#         annotated = image.copy()
#         for rice_type, confidence, (x, y, w, h) in results:
#             cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             cv2.putText(annotated, f"{rice_type} ({confidence:.2f})",
#                         (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         output_path = os.path.splitext(file_path)[0] + '_results.csv'
#         self.save_results(results, output_path)

#         annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
#         img_pil = Image.fromarray(annotated_rgb)
#         img_pil = img_pil.resize((400, 400))
#         img_tk = ImageTk.PhotoImage(img_pil)

#         if self.image_label is None:
#             self.image_label = Label(self.canvas, image=img_tk)
#             self.image_label.image = img_tk
#             self.image_label.pack()
#         else:
#             self.image_label.configure(image=img_tk)
#             self.image_label.image = img_tk

#     def setup_gui(self):
#         self.canvas = Canvas(self.root, width=400, height=500)
#         self.canvas.pack()

#         upload_button = Button(self.root, text="Upload Image", command=self.process_image)
#         upload_button.pack(pady=10)

#         Label(self.root, text="Rice Grain Classifier").pack(pady=10)

#     def run(self):
#         self.root.mainloop()

# if __name__ == "__main__":
#     app = RiceGrainClassifier()
#     app.run()











# import cv2
# import numpy as np
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from skimage.feature import hog

# class RiceGrainClassifierCamera:
#     def __init__(self):
#         self.model = None
#         self.scaler = None
#         self.rice_types = [
#             'Aromatic Heirloom Varieties',
#             'Deepwater Rice',
#             'Neang Minh',
#             'Phka Romduol',
#             'Phka Rumdeng'
#         ]
#         self.train_model()

#     def extract_features(self, grain_image):
#         grain_image = cv2.resize(grain_image, (64, 64))
#         gray = cv2.cvtColor(grain_image, cv2.COLOR_BGR2GRAY)

#         hog_features = hog(
#             gray, orientations=8, pixels_per_cell=(16, 16),
#             cells_per_block=(1, 1), visualize=False, feature_vector=True
#         )

#         contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if contours:
#             contour = max(contours, key=cv2.contourArea)
#             area = cv2.contourArea(contour)
#             perimeter = cv2.arcLength(contour, True)
#             circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
#         else:
#             area, perimeter, circularity = 0, 0, 0

#         color_mean = np.mean(grain_image, axis=(0, 1))
#         features = np.concatenate([hog_features, [area, perimeter, circularity], color_mean])

#         if len(features) != 134:
#             raise ValueError(f"Invalid feature length: {len(features)}")
#         return features

#     def preprocess_frame(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         denoised = cv2.fastNlMeansDenoising(gray)
#         thresh = cv2.adaptiveThreshold(
#             denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY_INV, 11, 2
#         )
#         return thresh

#     def detect_grains(self, frame, thresh):
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         grains = []
#         for contour in contours:
#             if cv2.contourArea(contour) > 50:
#                 x, y, w, h = cv2.boundingRect(contour)
#                 grain = frame[y:y + h, x:x + w]
#                 grains.append((grain, (x, y, w, h)))
#         return grains

#     def train_model(self):
#         X, y = [], []
#         for i in range(len(self.rice_types)):
#             for _ in range(20):
#                 features = np.random.rand(134)
#                 X.append(features)
#                 y.append(i)

#         self.scaler = StandardScaler()
#         X_scaled = self.scaler.fit_transform(X)
#         self.model = SVC(kernel='rbf', probability=True)
#         self.model.fit(X_scaled, y)

#     def classify_grains(self, grains):
#         results = []
#         for grain, (x, y, w, h) in grains:
#             try:
#                 features = self.extract_features(grain)
#                 features_scaled = self.scaler.transform([features])
#                 pred = self.model.predict(features_scaled)[0]
#                 confidence = np.max(self.model.predict_proba(features_scaled)[0])
#                 results.append((self.rice_types[pred], confidence, (x, y, w, h)))
#             except Exception as e:
#                 print("Skipping grain due to error:", e)
#         return results

#     def run_camera(self):
#         cap = cv2.VideoCapture(0)  # Use webcam (index 0)

#         if not cap.isOpened():
#             print("❌ Cannot open webcam.")
#             return

#         print("✅ Webcam started. Press 'q' to quit.")

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("⚠️ Failed to grab frame.")
#                 break

#             thresh = self.preprocess_frame(frame)
#             grains = self.detect_grains(frame, thresh)
#             results = self.classify_grains(grains)

#             for rice_type, confidence, (x, y, w, h) in results:
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 label = f"{rice_type} ({confidence:.2f})"
#                 cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
#                             0.5, (0, 255, 0), 2)

#             cv2.imshow("Rice Grain Detection (Camera)", frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()
#         print("✅ Webcam closed.")

# if __name__ == "__main__":
#     app = RiceGrainClassifierCamera()
#     app.run_camera()























# import cv2
# import numpy as np
# import pandas as pd
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from skimage.feature import hog
# from tkinter import Tk, Label, Button, filedialog, Canvas
# from PIL import Image, ImageTk
# import os

# class RiceGrainClassifier:
#     def __init__(self):
#         self.model = None
#         self.scaler = None
#         self.rice_types = [
#             'Aromatic Heirloom Varieties', 'Deepwater Rice', 'Neang Minh', 'Phka Romduol',
#             'Phka Rumdeng', 'Jasmine Rice', 'Glutinous', 'White Rice',
#             'Red Rice', 'Black Rice', 'Organic', 'Flooded Rice'
#         ]
#         self.root = Tk()
#         self.root.title("Rice Grain Classifier")
#         self.canvas = None
#         self.image_label = None
#         self.status_label = None
#         self.setup_gui()

#     def extract_features(self, grain_image):
#         # Resize grain image to fixed size
#         grain_image = cv2.resize(grain_image, (64, 64))

#         # Convert to grayscale
#         gray = cv2.cvtColor(grain_image, cv2.COLOR_BGR2GRAY)

#         # Extract HOG features
#         hog_features = hog(gray, orientations=8, pixels_per_cell=(16, 16),
#                            cells_per_block=(1, 1), visualize=False, feature_vector=True)

#         # Extract shape features
#         contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if contours:
#             contour = max(contours, key=cv2.contourArea)
#             area = cv2.contourArea(contour)
#             perimeter = cv2.arcLength(contour, True)
#             circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
#         else:
#             area, perimeter, circularity = 0, 0, 0

#         # Extract color features (mean RGB)
#         color_mean = np.mean(grain_image, axis=(0, 1))

#         features = np.concatenate([hog_features, [area, perimeter, circularity], color_mean])

#         assert len(features) == 134, f"Feature length is {len(features)} instead of 134"

#         return features

#     def preprocess_image(self, image_path):
#         image = cv2.imread(image_path)
#         if image is None:
#             return None, None

#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         denoised = cv2.fastNlMeansDenoising(gray)
#         thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                        cv2.THRESH_BINARY_INV, 11, 2)
#         return image, thresh

#     def detect_grains(self, image, thresh):
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         grains = []
#         for contour in contours:
#             if cv2.contourArea(contour) > 50:
#                 x, y, w, h = cv2.boundingRect(contour)
#                 grain = image[y:y+h, x:x+w]
#                 grains.append((grain, (x, y, w, h)))
#         return grains

#     def train_model(self):
#         # Simulated training data with correct feature length 134
#         X, y = [], []
#         for i, rice_type in enumerate(self.rice_types):
#             for _ in range(20):
#                 features = np.random.rand(134)  # Match extract_features length
#                 X.append(features)
#                 y.append(i)

#         self.scaler = StandardScaler()
#         X_scaled = self.scaler.fit_transform(X)

#         self.model = SVC(kernel='rbf', probability=True)
#         self.model.fit(X_scaled, y)

#     def classify_grains(self, grains):
#         if self.model is None:
#             self.train_model()

#         results = []
#         for grain, (x, y, w, h) in grains:
#             features = self.extract_features(grain)
#             features_scaled = self.scaler.transform([features])
#             pred = self.model.predict(features_scaled)[0]
#             confidence = np.max(self.model.predict_proba(features_scaled)[0])
#             results.append((self.rice_types[pred], confidence, (x, y, w, h)))

#         return results

#     def save_results(self, results, output_path):
#         df = pd.DataFrame([(r[0], r[1], r[2][0], r[2][1], r[2][2], r[2][3])
#                            for r in results],
#                           columns=['Rice_Type', 'Confidence', 'X', 'Y', 'Width', 'Height'])
#         df.to_csv(output_path, index=False)

#         counts = df['Rice_Type'].value_counts()
#         with open(os.path.splitext(output_path)[0] + '_summary.txt', 'w') as f:
#             f.write("Rice Type Counts:\n")
#             for rice_type, count in counts.items():
#                 f.write(f"{rice_type}: {count}\n")

#     def process_image(self):
#         file_path = filedialog.askopenfilename()
#         if not file_path:
#             self.status_label.config(text="No file selected.")
#             return

#         image, thresh = self.preprocess_image(file_path)
#         if image is None:
#             self.status_label.config(text="Failed to load image.")
#             return

#         grains = self.detect_grains(image, thresh)
#         if not grains:
#             self.status_label.config(text="No rice grains detected in image.")
#             return

#         results = self.classify_grains(grains)

#         annotated = image.copy()
#         for rice_type, confidence, (x, y, w, h) in results:
#             cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             cv2.putText(annotated, f"{rice_type} ({confidence:.2f})",
#                         (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         output_path = os.path.splitext(file_path)[0] + '_results.csv'
#         self.save_results(results, output_path)

#         annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
#         img_pil = Image.fromarray(annotated_rgb)
#         img_pil = img_pil.resize((400, 400))
#         img_tk = ImageTk.PhotoImage(img_pil)

#         if self.image_label is None:
#             self.image_label = Label(self.canvas, image=img_tk)
#             self.image_label.image = img_tk
#             self.image_label.pack()
#         else:
#             self.image_label.configure(image=img_tk)
#             self.image_label.image = img_tk

#         self.status_label.config(text=f"Classified {len(results)} grains. Results saved to:\n{output_path}")

#     def setup_gui(self):
#         self.canvas = Canvas(self.root, width=400, height=500)
#         self.canvas.pack()

#         upload_button = Button(self.root, text="Upload Image", command=self.process_image)
#         upload_button.pack(pady=10)

#         Label(self.root, text="Rice Grain Classifier").pack(pady=10)

#         self.status_label = Label(self.root, text="", fg="blue", wraplength=400)
#         self.status_label.pack(pady=5)

#     def run(self):
#         self.root.mainloop()

    
# if __name__ == "__main__":
#     app = RiceGrainClassifier()
#     app.run()







# import cv2
# import numpy as np
# import pandas as pd
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from skimage.feature import hog
# from tkinter import Tk, Label, Button, filedialog, Canvas
# from PIL import Image, ImageTk
# import os
# def classify_grains(self, grains):
#     if self.model is None:
#         self.train_model()

#     results = []
#     for grain, (x, y, w, h) in grains:
#         features = self.extract_features(grain)
#         features_scaled = self.scaler.transform([features])
#         pred = self.model.predict(features_scaled)[0]
#         confidence = np.max(self.model.predict_proba(features_scaled)[0])
#         results.append((self.rice_types[pred], confidence, (x, y, w, h)))
#     return results

# def process_image(self):
#     file_path = filedialog.askopenfilename()
#     if not file_path:
#         self.status_label.config(text="No file selected.")
#         return

#     image, thresh = self.preprocess_image(file_path)
#     if image is None:
#         self.status_label.config(text="Failed to load image.")
#         return

#     grains = self.detect_grains(image, thresh)
#     if not grains:
#         self.status_label.config(text="No rice grains detected in image.")
#         return

#     results = self.classify_grains(grains)

#     annotated = image.copy()
#     for rice_type, confidence, (x, y, w, h) in results:
#         cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         # Draw rice type name bigger and clearer
#         cv2.putText(annotated, f"{rice_type} ({confidence:.2f})",
#                     (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     output_path = os.path.splitext(file_path)[0] + '_results.csv'
#     self.save_results(results, output_path)

#     # Count each rice type and find the most common
#     rice_type_counts = pd.Series([r[0] for r in results]).value_counts()
#     most_common_rice = rice_type_counts.idxmax()
#     most_common_count = rice_type_counts.max()

#     annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
#     img_pil = Image.fromarray(annotated_rgb)
#     img_pil = img_pil.resize((400, 400))
#     img_tk = ImageTk.PhotoImage(img_pil)

#     if self.image_label is None:
#         self.image_label = Label(self.canvas, image=img_tk)
#         self.image_label.image = img_tk
#         self.image_label.pack()
#     else:
#         self.image_label.configure(image=img_tk)
#         self.image_label.image = img_tk

#     self.status_label.config(
#         text=(
#             f"Classified {len(results)} grains.\n"
#             f"🟢 Most Common Rice Type: {most_common_rice} ({most_common_count} grains)\n"
#             f"📄 Results saved to: {output_path}"
#         )
#     )
# class RiceGrainClassifier:
#     def __init__(self):
#         self.model = None
#         self.scaler = None
#         self.rice_types = [
#             'Aromatic Heirloom Varieties', 'Deepwater Rice', 'Neang Minh', 'Phka Romduol',
#             'Phka Rumdeng', 'Jasmine Rice', 'Glutinous', 'White Rice',
#             'Red Rice', 'Black Rice', 'Organic', 'Flooded Rice'
#         ]
#         self.root = Tk()
#         self.root.title("Rice Grain Classifier")
#         self.canvas = None
#         self.image_label = None
#         self.status_label = None
#         self.setup_gui()
    
#     def extract_features(self, grain_image):
#         grain_image = cv2.resize(grain_image, (64, 64))
#         gray = cv2.cvtColor(grain_image, cv2.COLOR_BGR2GRAY)

#         hog_features = hog(gray, orientations=8, pixels_per_cell=(16, 16),
#                            cells_per_block=(1, 1), visualize=False, feature_vector=True)

#         contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if contours:
#             contour = max(contours, key=cv2.contourArea)
#             area = cv2.contourArea(contour)
#             perimeter = cv2.arcLength(contour, True)
#             circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
#         else:
#             area, perimeter, circularity = 0, 0, 0

#         color_mean = np.mean(grain_image, axis=(0, 1))
#         features = np.concatenate([hog_features, [area, perimeter, circularity], color_mean])

#         assert len(features) == 134, f"Feature length is {len(features)} instead of 134"
#         return features

#     def preprocess_image(self, image_path):
#         image = cv2.imread(image_path)
#         if image is None:
#             return None, None

#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         denoised = cv2.fastNlMeansDenoising(gray)
#         thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                        cv2.THRESH_BINARY_INV, 11, 2)
#         return image, thresh

#     def detect_grains(self, image, thresh):
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         grains = []
#         for contour in contours:
#             if cv2.contourArea(contour) > 50:
#                 x, y, w, h = cv2.boundingRect(contour)
#                 grain = image[y:y+h, x:x+w]
#                 grains.append((grain, (x, y, w, h)))
#         return grains

#     def train_model(self):
#         # Simulated training data
#         X, y = [], []
#         for i, rice_type in enumerate(self.rice_types):
#             for _ in range(20):
#                 features = np.random.rand(134)
#                 X.append(features)
#                 y.append(i)

#         self.scaler = StandardScaler()
#         X_scaled = self.scaler.fit_transform(X)

#         self.model = SVC(kernel='rbf', probability=True)
#         self.model.fit(X_scaled, y)

#     def classify_grains(self, grains):
#         if self.model is None:
#             self.train_model()

#         results = []
#         for grain, (x, y, w, h) in grains:
#             features = self.extract_features(grain)
#             features_scaled = self.scaler.transform([features])
#             pred = self.model.predict(features_scaled)[0]
#             confidence = np.max(self.model.predict_proba(features_scaled)[0])
#             results.append((self.rice_types[pred], confidence, (x, y, w, h)))
#         return results

#     def save_results(self, results, output_path):
#         df = pd.DataFrame([(r[0], r[1], r[2][0], r[2][1], r[2][2], r[2][3])
#                            for r in results],
#                           columns=['Rice_Type', 'Confidence', 'X', 'Y', 'Width', 'Height'])
#         df.to_csv(output_path, index=False)

#         counts = df['Rice_Type'].value_counts()
#         with open(os.path.splitext(output_path)[0] + '_summary.txt', 'w') as f:
#             f.write("Rice Type Counts:\n")
#             for rice_type, count in counts.items():
#                 f.write(f"{rice_type}: {count}\n")

#     def process_image(self):
#         file_path = filedialog.askopenfilename()
#         if not file_path:
#             self.status_label.config(text="No file selected.")
#             return

#         image, thresh = self.preprocess_image(file_path)
#         if image is None:
#             self.status_label.config(text="Failed to load image.")
#             return

#         grains = self.detect_grains(image, thresh)
#         if not grains:
#             self.status_label.config(text="No rice grains detected in image.")
#             return

#         results = self.classify_grains(grains)

#         annotated = image.copy()
#         for rice_type, confidence, (x, y, w, h) in results:
#             cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             cv2.putText(annotated, f"{rice_type} ({confidence:.2f})",
#                         (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

#         output_path = os.path.splitext(file_path)[0] + '_results.csv'
#         self.save_results(results, output_path)

#         # Count and display most common rice type
#         rice_type_counts = pd.Series([r[0] for r in results]).value_counts()
#         most_common_rice = rice_type_counts.idxmax()
#         most_common_count = rice_type_counts.max()

#         # Convert image to display in GUI
#         annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
#         img_pil = Image.fromarray(annotated_rgb)
#         img_pil = img_pil.resize((400, 400))
#         img_tk = ImageTk.PhotoImage(img_pil)

#         if self.image_label is None:
#             self.image_label = Label(self.canvas, image=img_tk)
#             self.image_label.image = img_tk
#             self.image_label.pack()
#         else:
#             self.image_label.configure(image=img_tk)
#             self.image_label.image = img_tk

#         self.status_label.config(
#             text=(
#                 f"Classified {len(results)} grains.\n"
#                 f"🟢 Most Common Rice Type: {most_common_rice} ({most_common_count} grains)\n"
#                 f"📄 Results saved to: {output_path}"
#             )
#         )

#     def setup_gui(self):
#         self.canvas = Canvas(self.root, width=400, height=500)
#         self.canvas.pack()

#         upload_button = Button(self.root, text="Upload Image", command=self.process_image)
#         upload_button.pack(pady=10)

#         Label(self.root, text="Rice Grain Classifier").pack(pady=10)

#         self.status_label = Label(self.root, text="", fg="blue", wraplength=400)
#         self.status_label.pack(pady=5)

#     def run(self):
#         self.root.mainloop()

    
# if __name__ == "__main__":
#     app = RiceGrainClassifier()
#     app.run()




import cv2, os
import numpy as np, pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from tkinter import Tk, Label, Button, filedialog, Canvas
from PIL import Image, ImageTk

class RiceGrainClassifier:
    def __init__(self):
        self.model, self.scaler, self.image_label = None, None, None
        self.rice_types = ['Aromatic Heirloom Varieties', 'Deepwater Rice', 'Neang Minh',
                           'Phka Romduol', 'Phka Rumdeng', 'Jasmine Rice', 'Glutinous',
                           'White Rice', 'Red Rice', 'Black Rice', 'Organic', 'Flooded Rice']
        self.root = Tk(); self.root.title("Rice Grain Classifier")
        self.canvas = Canvas(self.root, width=400, height=500); self.canvas.pack()
        Button(self.root, text="Upload Image", command=self.process_image).pack(pady=10)
        Label(self.root, text="Rice Grain Classifier").pack(pady=10)
        self.status_label = Label(self.root, text="", fg="blue", wraplength=400); self.status_label.pack(pady=5)

    def extract_features(self, img):
        img = cv2.resize(img, (64, 64)); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog_feat = hog(gray, orientations=8, pixels_per_cell=(16,16), cells_per_block=(1,1), visualize=False)
        cnts, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            area, peri = cv2.contourArea(c), cv2.arcLength(c, True)
            circ = 4*np.pi*area/(peri*peri) if peri else 0
        else: area = peri = circ = 0
        color = np.mean(img, axis=(0,1))
        feats = np.concatenate([hog_feat, [area, peri, circ], color])
        assert len(feats) == 134, f"Expected 134 features, got {len(feats)}"
        return feats

    def preprocess_image(self, path):
        img = cv2.imread(path); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        return img, thresh

    def detect_grains(self, img, thresh):
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [(img[y:y+h, x:x+w], (x,y,w,h)) for c in cnts if cv2.contourArea(c)>50
                for x,y,w,h in [cv2.boundingRect(c)]]

    def train_model(self):
        X = [np.random.rand(134) for _ in range(240)]
        y = [i for i in range(12) for _ in range(20)]
        self.scaler = StandardScaler(); X_scaled = self.scaler.fit_transform(X)
        self.model = SVC(kernel='rbf', probability=True); self.model.fit(X_scaled, y)

    def classify_grains(self, grains):
        if not self.model: self.train_model()
        results = []
        for g, (x,y,w,h) in grains:
            feats = self.extract_features(g)
            scaled = self.scaler.transform([feats])
            pred = self.model.predict(scaled)[0]
            conf = np.max(self.model.predict_proba(scaled))
            results.append((self.rice_types[pred], conf, (x,y,w,h)))
        return results

    def save_results(self, results, out_path):
        df = pd.DataFrame([(r[0], r[1], *r[2]) for r in results],
                          columns=['Rice_Type','Confidence','X','Y','Width','Height'])
        df.to_csv(out_path, index=False)
        with open(out_path.replace('.csv', '_summary.txt'), 'w') as f:
            for k,v in df['Rice_Type'].value_counts().items(): f.write(f"{k}: {v}\n")

    def process_image(self):
        path = filedialog.askopenfilename()
        if not path: return self.status_label.config(text="No file selected.")
        img, thresh = self.preprocess_image(path)
        if img is None: return self.status_label.config(text="Failed to load image.")
        grains = self.detect_grains(img, thresh)
        if not grains: return self.status_label.config(text="No grains detected.")
        results = self.classify_grains(grains)
        out_csv = path.replace('.jpg','_results.csv')
        self.save_results(results, out_csv)

        annotated = img.copy()
        for name, conf, (x,y,w,h) in results:
            cv2.rectangle(annotated, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(annotated, f"{name} ({conf:.2f})", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(rgb).resize((400, 400)))
        if not self.image_label:
            self.image_label = Label(self.canvas, image=img_tk); self.image_label.image = img_tk
            self.image_label.pack()
        else:
            self.image_label.configure(image=img_tk); self.image_label.image = img_tk

        counts = pd.Series([r[0] for r in results]).value_counts()
        self.status_label.config(
            text=f"Classified {len(results)} grains.\n"
                 f" Most Common: {counts.idxmax()} ({counts.max()} grains)\n"
                 f"📄 Results saved to: {out_csv}"
        )

    def run(self): self.root.mainloop()

if __name__ == "__main__":
    RiceGrainClassifier().run()
