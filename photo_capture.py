import streamlit as st
from openai import OpenAI
from PIL import Image
import io
import base64
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

def extract_form_data(image_bytes):
    try:
        base64_image = encode_image(image_bytes)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                            "Extract donor details from this donation form:\n"
                            "1. Full Name\n2. Phone\n3. Email\n4. Donation Amount (number only)\n"
                            "Return exactly this format:\n"
                            "Full Name: [name]\nPhone: [phone]\nEmail: [email]\nAmount: [number]"
                        )},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }}
                    ]
                }
            ],
            max_tokens=200
        )

        content = response.choices[0].message.content.strip()
        data = {"Full Name": "", "Phone": "", "Email": "", "Amount": ""}

        for line in content.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key, value = key.strip(), value.strip()
                if key in data:
                    data[key] = value

        return data

    except Exception as e:
        st.error(f"Error reading form: {e}")
        return None

# --- Main Input Section ---
if "donors" not in st.session_state:
    st.session_state["donors"] = []
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
    image_bytes = buffered.getvalue()  # ✅ Get bytes from the buffer

with st.spinner("Analyzing photo..."):
    result = extract_form_data(image_bytes)
    if result:
        # Add to donor list
        st.session_state["donors"].append(result)

        st.success("✅ Donation form processed!")
        st.markdown(f"**Name:** {result['Full Name'] or 'Anonymous'}")
        st.markdown(f"**Amount:** ${result['Amount'] or '0'}")
    else:
        st.error("Could not extract donor details.")

# --- Live Donor Wall ---

if st.session_state["donors"]:
    st.markdown("## 🙌 Live Donor Wall")

    for donor in reversed(st.session_state["donors"]):
        name = donor["Full Name"] or "Anonymous"
        amount = donor["Amount"] or "0"
        st.markdown(f"🎁 **{name}** — ${amount}")

    # # 4️⃣ Send to OpenAI (OCR-style)
    # with st.spinner("Extracting donor name..."):
    #     response = client.chat.completions.create(
    #         model="gpt-4o-mini",
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "text", "text": "Extract the donor's name from this form image."},
    #                     {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_base64}"}
    #                 ]
    #             }
    #         ]
    #     )
    #     donor_name = response.choices[0].message.content

    # # 5️⃣ Show result
    # st.success(f"🧾 Donor name detected: **{donor_name}**")
