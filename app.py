import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import uuid  # random id generator
from streamlit_option_menu import option_menu
import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import datetime
import hashlib
import tensorflow as tf
import pickle
from pymongo import MongoClient, ASCENDING
from bson import Binary
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)

# MongoDB connection
# client = MongoClient('mongodb://localhost:27017/')
db = client['visitor_database']
visitors_collection = db['visitors']
history_collection = db['visitor_history']

# Load models and encoders
liveness_model = tf.keras.models.load_model('livenessnet_model.h5')
le = pickle.loads(open('label_encoder.pickle', 'rb').read())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device, keep_all=True)

# Define constants
COLOR_DARK = (255, 255, 255)
COLOR_WHITE = (75, 110, 192)
allowed_image_type = ['.png', 'jpg', '.jpeg']

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def initialize_data():
    try:
        visitors_collection.create_index([("Name", ASCENDING)], unique=True)
    except Exception as e:
        print(f"Index creation error: {e}")
    return visitors_collection

def add_data_db(visitor_details):
    try:
        visitors_collection.insert_one(visitor_details)
        st.success('Details Added Successfully!')
    except Exception as e:
        st.error(e)

def BGR_to_RGB(image_in_array):
    return cv2.cvtColor(image_in_array, cv2.COLOR_BGR2RGB)

def save_image(image_array):
    _, buffer = cv2.imencode('.jpg', image_array)
    return Binary(buffer.tobytes())

def attendance(visitor_id, name, img_binary, logout_time=None):
    now = datetime.datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    history_entry = {
        "id": Binary.from_uuid(visitor_id),
        "visitor_name": name,
        "login_time": dtString,
        "logout_time": logout_time,
        "img_binary": img_binary
    }
    history_collection.insert_one(history_entry)

def update_logout_time(username):
    now = datetime.datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    history_collection.update_one(
        {"visitor_name": username, "logout_time": None},
        {"$set": {"logout_time": dtString}}
    )

def can_login(username):
    latest_entry = history_collection.find_one({"visitor_name": username}, sort=[("login_time", -1)])
    return latest_entry is None or latest_entry["logout_time"] is not None

def send_email(to_email, subject, body):
    from_email = "fortifyauth@gmail.com"  # Replace with your email
    password = "byljptxgvcxswllm"  # Replace with your email password

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        st.success(f"Email sent to {to_email}")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

def generate_otp():
    return random.randint(100000, 999999)

def view_attendace(username):
    history_records = list(history_collection.find({"visitor_name": username}).sort("login_time", -1))
    df_attendance = pd.DataFrame(history_records, columns=["login_time", "logout_time", "visitor_name", "id"])

    if df_attendance.shape[0] > 0:
        view_all = st.checkbox('View All History')
        if view_all:
            st.write(df_attendance)
        else:
            page = st.number_input('Page Number', min_value=1, value=1, step=1)
            rows_per_page = 10
            start_row = (page - 1) * rows_per_page
            end_row = start_row + rows_per_page
            st.write(df_attendance[start_row:end_row])

        if st.button('Clear Database'):
            clear_user_history(username)
            view_attendace(username)

        selected_time = st.selectbox('Select Time to View Image:', df_attendance['login_time'])

        if selected_time:
            selected_record = history_collection.find_one({"visitor_name": username, "login_time": selected_time})
            if selected_record and 'img_binary' in selected_record:
                img_binary = selected_record['img_binary']
                img_array = np.frombuffer(img_binary, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                st.image(BGR_to_RGB(img))
    else:
        st.warning("No records found for this username.")

def clear_user_history(username):
    history_collection.delete_many({"visitor_name": username})
    st.success(f"History for {username} cleared successfully!")

def crop_image_with_ratio(img, height, width, middle):
    h, w = img.shape[:2]
    h = h - h % 4
    new_w = int(h / height) * width
    startx = middle - new_w // 2
    endx = middle + new_w // 2
    if startx <= 0:
        cropped_img = img[0:h, 0:new_w]
    elif endx >= w:
        cropped_img = img[0:h, w - new_w:w]
    else:
        cropped_img = img[0:h, startx:endx]
    return cropped_img


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-color: #756c83;
    background-size: 180%;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
}}
[data-testid ="stHeader"] {{
    background-color: #ECEFF4;         
}}
body{{
    color: #252630;
}}
a{{
    color: #000000;               
}}

</style>
"""
# page_bg_img = """
# <style>
# [data-testid="stAppViewContainer"] > .main {{
#     background-color: #756c83;
#     background-size: 180%;
#     background-position: top left;
#     background-repeat: no-repeat;
#     background-attachment: local;
# }}
# [data-testid ="stHeader"] {{
#     background-color: #756c83;
# }}
# body{{
#     color: #252630;
# }}
# a{{
#     color: #000000;
# }}
# </style>
# """

st.markdown(page_bg_img, unsafe_allow_html=True)

def main():
    if "otp_verified" not in st.session_state:
        st.session_state.otp_verified = False
    if "forgot_password_clicked" not in st.session_state:
        st.session_state.forgot_password_clicked = False
    if "login_session_active" not in st.session_state:
        st.session_state.login_session_active = False
    if "current_otp" not in st.session_state:
        st.session_state.current_otp = None
    if "otp_sent" not in st.session_state:
        st.session_state.otp_sent = False

    st.sidebar.image("F.png", width=250)

    menu_options = ["Home", "Add to Database", "Log In", "View Visitor History", "Log Out"]
    menu_icons = ["house", "person-plus", "person-check", "database", "door-closed"]

    with st.sidebar:
        option = option_menu(
            "Navigation",
            menu_options,
            icons=menu_icons,
            menu_icon="cast",
            default_index=0,
            orientation="vertical",
            key="main_navigation"
        )

    if option == "Home":
        st.markdown("""
        <div style="background-color:#756c83;padding:10px;border-radius:10px;display:flex;justify-content:space-between;align-items:center;">
            <h1 style="color:#fefefe;text-align:center;margin:auto 0;">Welcome to Biometric Login and Verification System!</h1>
            
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <h3 style="color:#fbfbfb;">
            Secure Your Access with Two-Factor Authentication
        </h3>
        <p style="color:#fbfbfb;">
            Welcome to our state-of-the-art two-factor authentication platform, where we use advanced facial recognition technology to enhance your login security. Our system is designed to provide seamless and secure access, combining the convenience of facial recognition with robust anti-spoofing measures.
        </p>
        <h3 style="color:#fbfbfb;">
            Why Choose Our Platform?
        </h3>
        <p style="color:#fbfbfb;">
            - Advanced Facial Recognition: Our cutting-edge facial recognition technology ensures quick and accurate identification.<br>
            - Anti-Spoofing Measures: We employ sophisticated anti-spoofing techniques to protect against fraudulent access attempts.<br>
            - Enhanced Security: By integrating facial recognition with two-factor authentication, we provide an extra layer of security for your accounts and transactions.
        </p>
        <p style="color:#fbfbfb;">
            Explore our website to learn more about how our platform safeguards your information and provides a secure, user-friendly authentication experience.
        </p>
        """, unsafe_allow_html=True)

        # st.markdown("""
        # <h3 style="color:#fbfbfb;">
        #     Key Features
        # </h3>
        # <div style="display:flex; justify-content:space-around; margin:20px 0;">
        #     <div style="text-align:center;">
        #         <img src="{image_path}" style="border-radius:50%;">
        #         <h3 style="color:#fbfbfb;">Scalability</h3>
        #         <p style="color:#fbfbfb;">Our system is designed to handle a large number of users seamlessly.</p>
        #     </div>
        #     <div style="text-align:center;">
        #         <img src="/Users/gauridubey/Desktop/what/shreya/face-recognition-attendance-anti-spoofing/cat.jpg" style="border-radius:50%;">
        #         <h3 style="color:#fbfbfb;">Relevance</h3>
        #         <p style="color:#fbfbfb;">Applicable across various industries including finance, education, and corporate sectors.</p>
        #     </div>
        # </div>
        # """, unsafe_allow_html=True)

        st.markdown("""
        <h3 style="color:#fbfbfb;">
            See it in Action!
        </h3>
        """, unsafe_allow_html=True)

        video_file = open('video.mp4', 'rb') #enter the filename with filepath

        video_bytes = video_file.read() #reading the file

        st.video(video_bytes) #displaying the video

        st.markdown("""<h3 style="color:#fbfbfb;">Learn More</h3>""", unsafe_allow_html=True)
        st.markdown("""
        <ul style="list-style-type:square;">
            <li style="color:#83afbd;"> <a href="https://drive.google.com/file/d/17nhP9LUk_10XWhagbd5q9D0SMK8V8oXL/view?usp=drive_link" target="_blank" style="color:#83afbd;text-decoration:none;">Project Documentation</a></li>
            <li style="color:#83afbd;"> <a href="https://github.com/khushisinha18/FortifyAuth-Biometric-System" target="_blank" style="color:#83afbd;text-decoration:none;">GitHub Repository</a></li>
            <li style="color:#83afbd;">
                <a href="mailto:fortifyauth@gmail.com" target="_blank" style="color:#83afbd;text-decoration:none;">
                    Contact Us
                </a>
            </li>
        </ul>
        """, unsafe_allow_html=True)

    elif option == 'Log In':
        st.markdown("<h3 style='color:#fbfbfb;'>Please enter your username:</h3>", unsafe_allow_html=True)
        entered_username = st.text_input("Username", "")

        if entered_username:
            if not can_login(entered_username):
                st.error("You are already logged in and need to log out before logging in again.")
                return

            visitor_id = uuid.uuid1()
            img_file_buffer = st.camera_input("Take a picture")

            if img_file_buffer is not None:
                bytes_data = img_file_buffer.getvalue()
                if bytes_data:
                    image_array = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    image_array_copy = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

                    st.success('Image Captured Successfully!')

                    max_faces = 0
                    rois = []
                    aligned = []
                    spoofs = []
                    can = []
                    face_locations, prob = mtcnn(image_array, return_prob=True)
                    boxes, _ = mtcnn.detect(image_array)

                    if boxes is not None:
                        boxes_int = boxes.astype(int)
                        spoofs = []
                        can = []

                        if face_locations is not None:
                            for idx, (left, top, right, bottom) in enumerate(boxes_int):
                                img = crop_image_with_ratio(image_array, 4, 3, (left + right) // 2)
                                face = cv2.resize(img, (32, 32)).astype('float') / 255.0
                                face = tf.keras.preprocessing.image.img_to_array(face)
                                face = np.expand_dims(face, axis=0)

                                preds = liveness_model.predict(face)[0]
                                j = np.argmax(preds)
                                label_name = le.classes_[j]

                                if label_name == 'real':
                                    spoofs.append("REAL")
                                    can.append(idx)
                                else:
                                    spoofs.append("FAKE")
                                    email_record = visitors_collection.find_one({"Name": entered_username})
                                    if email_record and "Email" in email_record:
                                        send_email(email_record["Email"], "Spoofing Attempt Detected", "A spoofing attempt was detected during login.")

                        for idx, (left, top, right, bottom) in enumerate(boxes_int):
                            rois.append(image_array[top:bottom, left:right].copy())
                            cv2.rectangle(image_array, (left, top), (right, bottom), COLOR_DARK, 2)
                            cv2.rectangle(image_array, (left, bottom + 35), (right, bottom), COLOR_DARK, cv2.FILLED)
                            font = cv2.FONT_HERSHEY_DUPLEX
                            cv2.putText(image_array, f"#{idx} {spoofs[idx]}", (left + 5, bottom + 25), font, .55,
                                        COLOR_WHITE, 1)

                        st.image(BGR_to_RGB(image_array), width=720)

                        max_faces = len(boxes_int)

                        if max_faces > 0:
                            col1, col2 = st.columns(2)
                            face_idxs = col1.multiselect("Select face#", can, default=can)

                            if len(face_idxs) > 0:
                                dataframe_new = pd.DataFrame()
                                aligned = []

                                for idx, loc in enumerate(face_locations):
                                    torch_loc = torch.stack([loc]).to(device)
                                    encodesCurFrame = resnet(torch_loc).detach().cpu()
                                    aligned.append(encodesCurFrame)

                                for face_idx in face_idxs:
                                    if spoofs[face_idx] == "FAKE":
                                        st.error("You are fake!")
                                    else:
                                        visitor_data = list(visitors_collection.find({}, {'_id': 0}))
                                        dataframe = pd.DataFrame(visitor_data)

                                        if len(aligned) < 1:
                                            st.error(f'Please Try Again for face#{face_idx}!')
                                        else:
                                            face_to_compare = aligned[face_idx].numpy()
                                            similarities = []
                                            for i in range(len(dataframe)):
                                                encodings = np.array([pickle.loads(dataframe.loc[i, f'v{j}']) for j in range(512)])
                                                similarities.append(np.linalg.norm(encodings - face_to_compare))
                                            dataframe['similarity'] = similarities
                                            dataframe['similarity'] = dataframe['similarity'].astype(float)

                                            dataframe_new = dataframe.drop_duplicates(keep='first')
                                            dataframe_new.reset_index(drop=True, inplace=True)
                                            dataframe_new.sort_values(by="similarity", ascending=True, inplace=True)
                                            dataframe_new = dataframe_new.head(1)
                                            dataframe_new.reset_index(drop=True, inplace=True)

                                            if dataframe_new.shape[0] > 0:
                                                (left, top, right, bottom) = (boxes_int[face_idx])

                                                rois.append(image_array[top:bottom, left:right].copy())
                                                cv2.rectangle(image_array, (left, top), (right, bottom), COLOR_DARK, 2)
                                                cv2.rectangle(image_array, (left, bottom + 35), (right, bottom), COLOR_DARK,
                                                              cv2.FILLED)
                                                font = cv2.FONT_HERSHEY_DUPLEX
                                                cv2.putText(image_array, f"#{dataframe_new.loc[0, 'Name']}",
                                                            (left + 5, bottom + 25), font, .55, COLOR_WHITE, 1)

                                                name_visitor = dataframe_new.loc[0, 'Name']
                                                email_visitor = dataframe_new.loc[0, 'Email']
                                                st.image(BGR_to_RGB(image_array), width=720)

                                                entered_password = st.text_input("Enter your password", type="password")
                                                forgot_password_clicked = st.button("Forgot Password")

                                                if entered_password:
                                                    hashed_entered_password = hash_password(entered_password)
                                                    stored_hashed_password = dataframe_new.loc[0, 'Password']
                                                    if hashed_entered_password == stored_hashed_password:
                                                        img_binary = save_image(image_array_copy)
                                                        attendance(visitor_id, name_visitor, img_binary)
                                                        st.session_state.login_session_active = True
                                                        st.success(f"Welcome {name_visitor}!")
                                                        st.session_state.otp_verified = False
                                                        st.session_state.forgot_password_clicked = False
                                                        st.session_state.otp_sent = False
                                                        st.session_state.current_otp = None
                                                        return

                                                    else:
                                                        st.error("Login Unsuccessful! Wrong Password")

                                                if forgot_password_clicked or st.session_state.forgot_password_clicked:
                                                    st.session_state.forgot_password_clicked = True
                                                    if not st.session_state.otp_sent:
                                                        otp = generate_otp()
                                                        st.session_state.current_otp = otp
                                                        send_email(email_visitor, "Password Reset OTP", f"Your OTP for password reset is {otp}")
                                                        st.success(f"OTP sent to {email_visitor}")
                                                        st.session_state.otp_sent = True

                                                    otp_input = st.text_input("Enter OTP", type="password")
                                                    if st.button("Verify OTP"):
                                                        if st.session_state.current_otp and otp_input.isnumeric() and int(otp_input) == st.session_state.current_otp:
                                                            st.session_state.otp_verified = True
                                                            st.success("OTP verified successfully. Please enter your new password.")
                                                        else:
                                                            st.error("Invalid OTP. Please try again.")

                                                    if st.session_state.otp_verified:
                                                        new_password = st.text_input("Enter new password", type="password")
                                                        confirm_password = st.text_input("Confirm new password", type="password")

                                                        if new_password and confirm_password and st.button("Change Password"):
                                                            if new_password == confirm_password:
                                                                new_hashed_password = hash_password(new_password)
                                                                visitors_collection.update_one({"Email": email_visitor}, {"$set": {"Password": new_hashed_password}})
                                                                st.success("Password reset successfully. Please log in again.")
                                                                st.session_state.otp_verified = False
                                                                st.session_state.forgot_password_clicked = False
                                                                st.session_state.otp_sent = False
                                                                st.session_state.current_otp = None
                                                                return
                                                            else:
                                                                st.error("Passwords do not match or are empty.")
                                            else:
                                                st.error(f'No Match Found for face#{face_idx}.')
                                                st.info('Please Update the database for a new person or click again!')
                                                img_binary = save_image(image_array_copy)
                                                attendance(visitor_id, 'Unknown', img_binary)
                        else:
                            st.error("No human face detected.")
                else:
                    st.error("Failed to capture image from camera.")
            else:
                st.info("Please take a picture using the camera.")

    elif option == 'Add to Database':
        col1, col2, col3 = st.columns(3)

        face_name = col1.text_input('Name:', '')
        face_password = col2.text_input('Password:', type='password')
        face_email = col3.text_input('Email:', '')
        pic_option = col2.radio('Upload Picture', options=["Upload a Picture", "Take a Picture with Cam"],
                                key="pic_option")

        if pic_option == 'Upload a Picture':
            img_file_buffer = col3.file_uploader('Upload a Picture', type=allowed_image_type)
            if img_file_buffer is not None:
                bytes_data = img_file_buffer.getvalue()
                file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)

        elif pic_option == 'Take a Picture with Cam':
            img_file_buffer = col3.camera_input("Take a Picture with Cam")
            if img_file_buffer is not None:
                bytes_data = img_file_buffer.getvalue()
                file_bytes = np.frombuffer(bytes_data, np.uint8)

        if 'bytes_data' in locals() and ((img_file_buffer is not None) & (len(face_name) > 1) & (len(face_password) > 1) & st.button('Click to Save!')):
            image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            face_locations, prob = mtcnn(image_array, return_prob=True)
            torch_loc = torch.stack([face_locations[0]]).to(device)
            encodesCurFrame = resnet(torch_loc).detach().cpu().numpy()

            visitor_details = {
                "Name": face_name,
                "Password": hash_password(face_password),
                "Email": face_email,
                "Image": bytes_data,  # Save the image data as binary
            }
            for i, encode in enumerate(encodesCurFrame[0]):
                visitor_details[f'v{i}'] = pickle.dumps(encode)  # Serialize the numpy array

            # Initialize or read the existing database
            initialize_data()
            add_data_db(visitor_details)  # Add the new data to the database

    elif option == "View Visitor History":
        username = st.text_input("Enter username to view history:")
        if username:
            view_attendace(username)

    elif option == "Log Out":
        st.markdown("<h3 style='color:#fbfbfb;'>Please confirm your identity to logout:</h3>", unsafe_allow_html=True)
        entered_username = st.text_input("Username", "")

        if entered_username:
            img_file_buffer = st.camera_input("Take a picture")

            if img_file_buffer is not None:
                bytes_data = img_file_buffer.getvalue()
                if bytes_data:
                    image_array = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

                    st.success('Image Captured Successfully!')

                    max_faces = 0
                    rois = []
                    aligned = []
                    spoofs = []
                    can = []
                    face_locations, prob = mtcnn(image_array, return_prob=True)
                    boxes, _ = mtcnn.detect(image_array)

                    if boxes is not None:
                        boxes_int = boxes.astype(int)
                        spoofs = []
                        can = []

                        if face_locations is not None:
                            for idx, (left, top, right, bottom) in enumerate(boxes_int):
                                img = crop_image_with_ratio(image_array, 4, 3, (left + right) // 2)
                                face = cv2.resize(img, (32, 32)).astype('float') / 255.0
                                face = tf.keras.preprocessing.image.img_to_array(face)
                                face = np.expand_dims(face, axis=0)

                                preds = liveness_model.predict(face)[0]
                                j = np.argmax(preds)
                                label_name = le.classes_[j]

                                if label_name == 'real':
                                    spoofs.append("REAL")
                                    can.append(idx)
                                else:
                                    spoofs.append("FAKE")
                                    email_record = visitors_collection.find_one({"Name": entered_username})
                                    if email_record and "Email" in email_record:
                                        send_email(email_record["Email"], "Spoofing Attempt Detected", "A spoofing attempt was detected during logout.")

                        for idx, (left, top, right, bottom) in enumerate(boxes_int):
                            rois.append(image_array[top:bottom, left:right].copy())
                            cv2.rectangle(image_array, (left, top), (right, bottom), COLOR_DARK, 2)
                            cv2.rectangle(image_array, (left, bottom + 35), (right, bottom), COLOR_DARK, cv2.FILLED)
                            font = cv2.FONT_HERSHEY_DUPLEX
                            cv2.putText(image_array, f"#{idx} {spoofs[idx]}", (left + 5, bottom + 25), font, .55,
                                        COLOR_WHITE, 1)

                        st.image(BGR_to_RGB(image_array), width=720)

                        max_faces = len(boxes_int)

                        if max_faces > 0:
                            col1, col2 = st.columns(2)
                            face_idxs = col1.multiselect("Select face#", can, default=can)

                            if len(face_idxs) > 0:
                                aligned = []
                                for idx, loc in enumerate(face_locations):
                                    torch_loc = torch.stack([loc]).to(device)
                                    encodesCurFrame = resnet(torch_loc).detach().cpu()
                                    aligned.append(encodesCurFrame)

                                for face_idx in face_idxs:
                                    if spoofs[face_idx] == "FAKE":
                                        st.error("You are fake!")
                                    else:
                                        visitor_data = list(visitors_collection.find({}, {'_id': 0}))
                                        dataframe = pd.DataFrame(visitor_data)

                                        if len(aligned) < 1:
                                            st.error(f'Please Try Again for face#{face_idx}!')
                                        else:
                                            face_to_compare = aligned[face_idx].numpy()
                                            similarities = []
                                            for i in range(len(dataframe)):
                                                encodings = np.array([pickle.loads(dataframe.loc[i, f'v{j}']) for j in range(512)])
                                                similarities.append(np.linalg.norm(encodings - face_to_compare))
                                            dataframe['similarity'] = similarities
                                            dataframe['similarity'] = dataframe['similarity'].astype(float)

                                            dataframe_new = dataframe.drop_duplicates(keep='first')
                                            dataframe_new.reset_index(drop=True, inplace=True)
                                            dataframe_new.sort_values(by="similarity", ascending=True, inplace=True)
                                            dataframe_new = dataframe_new.head(1)
                                            dataframe_new.reset_index(drop=True, inplace=True)

                                            if dataframe_new.shape[0] > 0:
                                                (left, top, right, bottom) = (boxes_int[face_idx])

                                                rois.append(image_array[top:bottom, left:right].copy())
                                                cv2.rectangle(image_array, (left, top), (right, bottom), COLOR_DARK, 2)
                                                cv2.rectangle(image_array, (left, bottom + 35), (right, bottom), COLOR_DARK,
                                                              cv2.FILLED)
                                                font = cv2.FONT_HERSHEY_DUPLEX
                                                cv2.putText(image_array, f"#{dataframe_new.loc[0, 'Name']}",
                                                            (left + 5, bottom + 25), font, .55, COLOR_WHITE, 1)

                                                name_visitor = dataframe_new.loc[0, 'Name']
                                                email_visitor = dataframe_new.loc[0, 'Email']
                                                st.image(BGR_to_RGB(image_array), width=720)

                                                entered_password = st.text_input("Enter your password", type="password")

                                                if entered_password:
                                                    hashed_entered_password = hash_password(entered_password)
                                                    stored_hashed_password = dataframe_new.loc[0, 'Password']
                                                    if hashed_entered_password == stored_hashed_password:
                                                        update_logout_time(name_visitor)
                                                        st.session_state.login_session_active = False
                                                        st.success("Logged out successfully!")
                                                        st.experimental_rerun()
                                                    else:
                                                        st.error("Logout failed! Incorrect password.")
                                            else:
                                                st.error(f'No Match Found for face#{face_idx}.')
                                                st.info('Please Try Again!')
                    else:
                        st.error("No human face detected.")
                else:
                    st.error("Failed to capture image from camera.")
            else:
                st.info("Please take a picture using the camera.")

if __name__ == '__main__':
    main()

