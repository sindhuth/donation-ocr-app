import streamlit as st
import base64
from openai import OpenAI
from io import BytesIO

# Streamlit page setup
st.set_page_config(
    page_title="📸 Live Donation Capture",
    page_icon="🎁",
    layout="centered"
)

st.title("📸 Live Donation Capture")
st.markdown("Take a photo of a **donation form**, and see donor details appear live below!")

# Store donors across uploads
if "donors" not in st.session_state:
    st.session_state["donors"] = []

# Sidebar for API key
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    st.info("This uses GPT-4o-mini for image-to-text OCR.")

if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.stop()

client = OpenAI(api_key=api_key)

# --- Helper Functions ---

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

uploaded_file = st.file_uploader(
    "📷 Take or upload a photo of the donation form",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    help="Use your phone camera to take a photo."
)

if uploaded_file is not None:
    with st.spinner("Analyzing photo..."):
        image_bytes = uploaded_file.read()
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

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit + OpenAI GPT-4o-mini")
