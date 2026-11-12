import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import time
import urllib.parse
import requests

st.set_page_config(
    page_title="Safety Helmet Vision AI",
    page_icon="⛑️",
    layout="wide"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "helmet_detector_best.pt")

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

def analyze_detections(result, model_names):
    boxes = result.boxes
    helmet_count = 0
    no_helmet_count = 0
    detections_list = []

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            class_name = model_names.get(cls_id, str(cls_id))
            if class_name == "helmet":
                helmet_count += 1
            elif class_name == "no_helmet":
                no_helmet_count += 1
            detections_list.append({
                "Class": class_name,
                "Confidence": f"{conf_val:.2%}"
            })
    return helmet_count, no_helmet_count, detections_list

def send_callmebot_whatsapp(phone, apikey, message):
    try:
        clean_phone = phone.strip().replace(" ", "")
        encoded_msg = urllib.parse.quote_plus(message)
        url = f"https://api.callmebot.com/whatsapp.php?phone={clean_phone}&text={encoded_msg}&apikey={apikey.strip()}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200 and "ERROR" not in resp.text.upper():
            return True, "WhatsApp alert sent via CallMeBot API."
        return False, f"CallMeBot API response ({resp.status_code}): {resp.text[:150]}"
    except Exception as e:
        return False, str(e)

def send_meta_cloud_whatsapp(phone_number_id, access_token, to_phone, message, use_template=False):
    try:
        clean_to = to_phone.strip().replace("+", "").replace(" ", "")
        url = f"https://graph.facebook.com/v26.0/{phone_number_id.strip()}/messages"
        headers = {
            "Authorization": f"Bearer {access_token.strip()}",
            "Content-Type": "application/json"
        }
        if use_template:
            payload = {
                "messaging_product": "whatsapp",
                "to": clean_to,
                "type": "template",
                "template": {
                    "name": "hello_world",
                    "language": {"code": "en_US"}
                }
            }
        else:
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": clean_to,
                "type": "text",
                "text": {"preview_url": False, "body": message}
            }
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        if resp.status_code in (200, 201):
            return True, "WhatsApp alert sent via Meta Cloud API."
        return False, f"Meta API Error ({resp.status_code}): {resp.text[:200]}"
    except Exception as e:
        return False, str(e)

def send_twilio_whatsapp(account_sid, auth_token, from_num, to_num, message):
    try:
        sid = account_sid.strip()
        token = auth_token.strip()
        clean_from = from_num.strip().replace("whatsapp:", "")
        clean_to = to_num.strip().replace("whatsapp:", "")
        url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
        payload = {
            "From": f"whatsapp:{clean_from}",
            "To": f"whatsapp:{clean_to}",
            "Body": message
        }
        resp = requests.post(url, data=payload, auth=(sid, token), timeout=10)
        if resp.status_code in (200, 201):
            return True, "WhatsApp alert sent via Twilio API."
        return False, f"Twilio API Error ({resp.status_code}): {resp.text[:200]}"
    except Exception as e:
        return False, str(e)

def dispatch_whatsapp_notification(message, force=False):
    current_time = time.time()
    cooldown = st.session_state.get("alert_cooldown", 30)
    last_sent = st.session_state.get("last_alert_time", 0)

    if not force and (current_time - last_sent < cooldown):
        return False, "Cooldown active"

    provider = st.session_state.get("wa_provider", "Disabled")
    success = False
    status_msg = "WhatsApp notifications are disabled."

    if provider == "CallMeBot WhatsApp API (Free & Easy)":
        phone = st.session_state.get("cmb_phone", "")
        apikey = st.session_state.get("cmb_apikey", "")
        if phone and apikey:
            success, status_msg = send_callmebot_whatsapp(phone, apikey, message)
        else:
            return False, "Enter both Manager WhatsApp Number and CallMeBot API Key."

    elif provider == "Meta WhatsApp Cloud API (Official)":
        phone_id = st.session_state.get("meta_phone_id", "")
        token = st.session_state.get("meta_token", "")
        to_phone = st.session_state.get("meta_to_phone", "")
        use_tpl = st.session_state.get("meta_use_template", False)
        if phone_id and token and to_phone:
            success, status_msg = send_meta_cloud_whatsapp(phone_id, token, to_phone, message, use_tpl)
        else:
            return False, "Enter Phone Number ID, Access Token, and Manager WhatsApp Number."

    elif provider == "Twilio WhatsApp API":
        sid = st.session_state.get("tw_sid", "")
        token = st.session_state.get("tw_token", "")
        from_num = st.session_state.get("tw_from", "")
        to_num = st.session_state.get("tw_to", "")
        if sid and token and from_num and to_num:
            success, status_msg = send_twilio_whatsapp(sid, token, from_num, to_num, message)
        else:
            return False, "Enter all Twilio WhatsApp credentials."

    if success:
        st.session_state["last_alert_time"] = current_time
        st.session_state["last_alert_status"] = f"✅ {status_msg} ({time.strftime('%H:%M:%S')})"
    else:
        st.session_state["last_alert_status"] = f"❌ {status_msg}"

    return success, status_msg

st.title("⛑️ Safety Helmet Vision AI - Industrial Safety Helmet Detection System")
st.markdown("**Team 11** | Automated real-time workplace safety compliance and WhatsApp violation alert system.")

st.sidebar.header("⚙️ Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.10, max_value=0.95, value=0.50, step=0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", min_value=0.10, max_value=0.95, value=0.45, step=0.05)
input_mode = st.sidebar.radio(
    "Select Input Source",
    ["Live Webcam (Auto-Start)", "Browser Camera Snapshot", "Image Upload", "Video Upload"]
)

st.sidebar.markdown("---")
st.sidebar.header("💬 WhatsApp API Configuration")
st.session_state["alert_cooldown"] = st.sidebar.number_input(
    "Alert Cooldown (seconds)", min_value=10, max_value=300, value=30
)

wa_provider = st.sidebar.selectbox(
    "Select WhatsApp Gateway",
    [
        "CallMeBot WhatsApp API (Free & Easy)",
        "Meta WhatsApp Cloud API (Official)",
        "Twilio WhatsApp API",
        "Disabled"
    ]
)
st.session_state["wa_provider"] = wa_provider

if wa_provider == "CallMeBot WhatsApp API (Free & Easy)":
    st.session_state["cmb_phone"] = st.sidebar.text_input(
        "Manager WhatsApp Number (with +code)", placeholder="+919876543210"
    )
    st.session_state["cmb_apikey"] = st.sidebar.text_input(
        "CallMeBot API Key", type="password", placeholder="e.g. 1234567"
    )
    st.sidebar.caption(
        "Save +34 621 062 163 on WhatsApp and send: I allow callmebot to send me messages"
    )

elif wa_provider == "Meta WhatsApp Cloud API (Official)":
    st.session_state["meta_phone_id"] = st.sidebar.text_input(
        "Meta Phone Number ID", placeholder="e.g. 105938475610293"
    )
    st.session_state["meta_token"] = st.sidebar.text_input(
        "Meta Bearer Access Token", type="password", placeholder="EAAG..."
    )
    st.session_state["meta_to_phone"] = st.sidebar.text_input(
        "Manager WhatsApp Number", placeholder="919876543210"
    )
    st.session_state["meta_use_template"] = st.sidebar.checkbox(
        "Use hello_world template (for first-time sandbox test)", value=False
    )

elif wa_provider == "Twilio WhatsApp API":
    st.session_state["tw_sid"] = st.sidebar.text_input("Twilio Account SID", placeholder="AC...")
    st.session_state["tw_token"] = st.sidebar.text_input("Twilio Auth Token", type="password")
    st.session_state["tw_from"] = st.sidebar.text_input(
        "Twilio WhatsApp Sandbox Number", value="+14155238886"
    )
    st.session_state["tw_to"] = st.sidebar.text_input(
        "Manager WhatsApp Number", placeholder="+919876543210"
    )

if wa_provider != "Disabled":
    if st.sidebar.button("📲 Send Test WhatsApp Alert", use_container_width=True):
        test_msg = "✅ Safety Helmet Vision AI: Your mobile is connected for real-time WhatsApp helmet violation alerts!"
        ok, info = dispatch_whatsapp_notification(test_msg, force=True)
        if ok:
            st.sidebar.success(info)
        else:
            st.sidebar.error(info)

if "last_alert_status" in st.session_state:
    st.sidebar.info(st.session_state["last_alert_status"])

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load YOLOv8 model: {e}")
    st.stop()

if input_mode == "Live Webcam (Auto-Start)":
    st.subheader("📹 Live Webcam Real-Time Helmet Monitoring")
    run_camera = st.toggle("Active Live Webcam Stream", value=True)

    m1, m2, m3 = st.columns(3)
    total_metric = m1.empty()
    helmet_metric = m2.empty()
    no_helmet_metric = m3.empty()

    alert_banner = st.empty()
    frame_window = st.empty()

    if run_camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.warning("Local webcam (Camera 0) is not accessible in this environment. Use Browser Camera below:")
            camera_photo = st.camera_input("Live Browser Camera")
            if camera_photo is not None:
                image = Image.open(camera_photo).convert("RGB")
                img_array = np.array(image)
                results = model.predict(source=img_array, conf=conf_threshold, iou=iou_threshold, verbose=False)
                result = results[0]
                annotated_rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
                helmet_count, no_helmet_count, _ = analyze_detections(result, model.names)

                total_metric.metric("Total Workers Detected", helmet_count + no_helmet_count)
                helmet_metric.metric("✅ Wearing Helmet", helmet_count)
                no_helmet_metric.metric("❌ NOT Wearing Helmet", no_helmet_count)

                if no_helmet_count > 0:
                    alert_text = f"🚨 SAFETY VIOLATION: {no_helmet_count} worker(s) detected NOT wearing a safety helmet!"
                    alert_banner.error(alert_text)
                    dispatch_whatsapp_notification(alert_text)
                elif helmet_count > 0:
                    alert_banner.success(f"✅ Safe: All {helmet_count} worker(s) are wearing helmets.")
                else:
                    alert_banner.info("No workers detected in frame.")

                frame_window.image(annotated_rgb, channels="RGB", use_container_width=True)
        else:
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Webcam stream ended or frame could not be read.")
                    break

                results = model.predict(source=frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
                result = results[0]
                annotated_frame = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

                helmet_count, no_helmet_count, _ = analyze_detections(result, model.names)

                total_metric.metric("Total Workers Detected", helmet_count + no_helmet_count)
                helmet_metric.metric("✅ Wearing Helmet", helmet_count)
                no_helmet_metric.metric("❌ NOT Wearing Helmet", no_helmet_count)

                if no_helmet_count > 0:
                    alert_text = f"🚨 SAFETY VIOLATION: {no_helmet_count} worker(s) detected NOT wearing a safety helmet!"
                    alert_banner.error(alert_text)
                    dispatch_whatsapp_notification(alert_text)
                elif helmet_count > 0:
                    alert_banner.success(f"✅ Safe: All {helmet_count} worker(s) are wearing helmets.")
                else:
                    alert_banner.info("Scanning live camera feed for workers...")

                frame_window.image(annotated_frame, channels="RGB", use_container_width=True)
                time.sleep(0.03)

            cap.release()

elif input_mode == "Browser Camera Snapshot":
    camera_photo = st.camera_input("Camera feed (works on both Local & Cloud servers)")
    if camera_photo is not None:
        image = Image.open(camera_photo).convert("RGB")
        img_array = np.array(image)

        results = model.predict(source=img_array, conf=conf_threshold, iou=iou_threshold)
        result = results[0]
        annotated_rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

        helmet_count, no_helmet_count, detections_list = analyze_detections(result, model.names)

        st.image(annotated_rgb, caption="Analyzed Frame", use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Workers Detected", helmet_count + no_helmet_count)
        m2.metric("✅ Wearing Helmet", helmet_count)
        m3.metric("❌ NOT Wearing Helmet", no_helmet_count)

        if no_helmet_count > 0:
            alert_text = f"🚨 SAFETY VIOLATION: {no_helmet_count} worker(s) detected NOT wearing a safety helmet!"
            st.error(alert_text)
            dispatch_whatsapp_notification(alert_text)
        elif helmet_count > 0:
            st.success(f"✅ Safe: All {helmet_count} worker(s) are wearing helmets.")
        else:
            st.warning("⚠️ No workers or helmets detected.")

elif input_mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload an industrial workplace image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        results = model.predict(source=img_array, conf=conf_threshold, iou=iou_threshold)
        result = results[0]
        annotated_rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

        helmet_count, no_helmet_count, detections_list = analyze_detections(result, model.names)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("Detection Result")
            st.image(annotated_rgb, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Workers Detected", helmet_count + no_helmet_count)
        m2.metric("✅ Wearing Helmet", helmet_count)
        m3.metric("❌ NOT Wearing Helmet", no_helmet_count)

        if no_helmet_count > 0:
            alert_text = f"🚨 SAFETY VIOLATION: {no_helmet_count} worker(s) detected NOT wearing a safety helmet!"
            st.error(alert_text)
            dispatch_whatsapp_notification(alert_text)
        elif helmet_count > 0:
            st.success(f"✅ Safe: All {helmet_count} detected worker(s) are wearing safety helmets.")
        else:
            st.warning("⚠️ No helmet or worker objects detected.")

        if detections_list:
            st.subheader("Detection Breakdown")
            st.dataframe(detections_list, use_container_width=True)

elif input_mode == "Video Upload":
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        m1, m2, m3 = st.columns(3)
        total_metric = m1.empty()
        helmet_metric = m2.empty()
        no_helmet_metric = m3.empty()

        st_alert = st.empty()
        st_frame = st.empty()
        progress_bar = st.progress(0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % 3 != 0:
                continue

            results = model.predict(source=frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
            result = results[0]
            annotated_frame = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
            helmet_count, no_helmet_count, _ = analyze_detections(result, model.names)

            total_metric.metric("Total Workers Detected", helmet_count + no_helmet_count)
            helmet_metric.metric("✅ Wearing Helmet", helmet_count)
            no_helmet_metric.metric("❌ NOT Wearing Helmet", no_helmet_count)

            st_frame.image(annotated_frame, caption=f"Frame {frame_idx}/{total_frames}", use_container_width=True)
            if no_helmet_count > 0:
                alert_text = f"🚨 SAFETY VIOLATION (Frame {frame_idx}): {no_helmet_count} worker(s) NOT wearing a helmet!"
                st_alert.error(alert_text)
                dispatch_whatsapp_notification(alert_text)
            elif helmet_count > 0:
                st_alert.success(f"✅ Compliant (Frame {frame_idx}): {helmet_count} helmet(s) detected.")

            progress_bar.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        os.unlink(tfile.name)
        st.success("Video processing complete.")
