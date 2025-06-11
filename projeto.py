import streamlit as st
import cv2
import numpy as np

st.title("Visão Computacional")

image = None
picture = None

uploaded_file = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file:
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

open_camera = st.button("Abrir camera")

if open_camera:
  # ip = st.text_input("Address")

  # if st.button("Connect"):
  #   cap = cv2.VideoCapture(ip)
    
  #   if not cap.isOpened():
  #     st.error("Error: Could not open video stream.")
  #   else:
  #     frame_placeholder = st.empty()
      
  #     stop_button = st.button("Disconnect")
  #     take_picture = st.button("Take Picture")
      
  #     while cap.isOpened() and not stop_button and not take_picture:
  #       ret, frame = cap.read()
  #       picture = frame
  #       if not ret:
  #         st.error("End of stream or error.")
  #         break

  #       frame_placeholder.image(image, caption="Camera")

  #     cap.release()
      
  #     st.image(picture)
  #     file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
  #     image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
  #     enable = False
  
  picture = st.camera_input("Take a picture", disabled=not open_camera)

if picture:
  st.image(picture)
  file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
  image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
  enable = False

if image is not None:
  st.subheader("Imagem Original")
  st.image(image, channels="RGB", width=400)
  altura, largura, canais = image.shape
  st.caption(f"Altura: {altura}; Largura: {largura}; Canais de cor: {canais}; Total de Pixels: {image.size}")

  st.subheader("Ajuste os thresholds do Canny")
  threshold1 = st.slider("Threshold 1", min_value=0, max_value=255, value=100)
  threshold2 = st.slider("Threshold 2", min_value=0, max_value=255, value=200)

  # Imagem cinza
  img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  st.subheader("Imagem Cinza")
  st.image(img_gray, width=400)
  altura_gray, largura_gray = img_gray.shape
  st.caption(f"Altura: {altura_gray}; Largura: {largura_gray}; Total de Pixels: {img_gray.size}")

  # Detecção de bordas
  edges = cv2.Canny(img_gray, threshold1, threshold2)

  st.subheader("Imagem com Detecção de Bordas")
  st.image(edges, caption="Bordas detectadas", channels="GRAY", width=400)
  
  st.subheader("Identificação de Objetos")
  if st.button("Identificar Objetos"):
    
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    blob = cv2.dnn.blobFromImage(img_gray, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    object_counts = {}
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5: # Consider detections with sufficient confidence
                label = classes[class_id]
                if label in object_counts:
                    object_counts[label] += 1
                else:
                    object_counts[label] = 1
    
    # Print object counts
    for label, count in object_counts.items():
        print(f"Number of {label}s: {count}")

    # image_objects = image
    
    # cv2.drawContours(image_objects, contours, -1, (0, 255, 0), 2)
    # for (x, y, w, h) in contours:
    #     cv2.rectangle(image_objects, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # st.image(image_objects, width=400)
  
  st.subheader("Identificação de Rostos")
  if st.button("Identificar Rostos"):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)

    image_faces = image

    for (x, y, w, h) in faces:
        cv2.rectangle(image_faces, (x, y), (x+w, y+h), (255, 0, 0), 2)
    st.image(image_faces, width=400)