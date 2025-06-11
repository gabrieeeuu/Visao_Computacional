import cv2
import streamlit as st

ip = st.text_input("Address")

if st.button("Conect"):
  cap = cv2.VideoCapture(ip)
  
  if not cap.isOpened():
    st.error("Error: Could not open video stream.")
  else:
    frame = None
    frame_placeholder = st.empty()
    
    stop_button = st.button("Stop")
    take_picture = st.button("Take Picture")
    
    while cap.isOpened() and not stop_button and not take_picture:
      ret, frame = cap.read()
      if not ret:
        st.error("End of stream or error.")
        break

      frame_placeholder.image(frame, caption="Camera")
      frame = frame

    cap.release()