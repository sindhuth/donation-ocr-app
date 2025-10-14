import streamlit as st
from openai import OpenAI
from PIL import Image
import io
import base64
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("📸 Live Donation Capture")

st.write("Take a photo of a donation form and extract the donor’s name.")

# 1️⃣ Capture image from camera
photo = st.camera_input("Take a photo of the donation form")

if photo:
    # 2️⃣ Display the photo
    image = Image.open(photo)
    st.image(image, caption="Captured Image", use_container_width=True)

    # 3️⃣ Convert to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    # 4️⃣ Send to OpenAI (OCR-style)
    with st.spinner("Extracting donor name..."):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the donor's name from this form image."},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_base64}"}
                    ]
                }
            ]
        )
        donor_name = response.choices[0].message.content

    # 5️⃣ Show result
    st.success(f"🧾 Donor name detected: **{donor_name}**")
