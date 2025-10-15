import streamlit as st
from openai import OpenAI
from PIL import Image
import io
import base64
import os
import sqlite3
from datetime import datetime
import threading
import pandas as pd
import time
from streamlit_extras.card import card

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Database lock for thread-safe operations
db_lock = threading.Lock()

def init_db():
    """Initialize SQLite database"""
    with db_lock:
        conn = sqlite3.connect('donors.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS donors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT,
                phone TEXT,
                email TEXT,
                amount TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS admin_session (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                session_id TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

def set_admin_session(session_id):
    """Set the admin session ID"""
    with db_lock:
        conn = sqlite3.connect('donors.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('INSERT OR REPLACE INTO admin_session (id, session_id) VALUES (1, ?)', (session_id,))
        conn.commit()
        conn.close()

def get_admin_session():
    """Get the admin session ID"""
    with db_lock:
        conn = sqlite3.connect('donors.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('SELECT session_id FROM admin_session WHERE id = 1')
        result = c.fetchone()
        conn.close()
        return result[0] if result else None

def add_donor(data):
    """Add donor to database"""
    with db_lock:
        conn = sqlite3.connect('donors.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('''
            INSERT INTO donors (full_name, phone, email, amount)
            VALUES (?, ?, ?, ?)
        ''', (data['Full Name'], data['Phone'], data['Email'], data['Amount']))
        conn.commit()
        conn.close()

def get_all_donors():
    """Get all donors from database"""
    with db_lock:
        conn = sqlite3.connect('donors.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('SELECT full_name, phone, email, amount, timestamp FROM donors ORDER BY timestamp DESC')
        donors = c.fetchall()
        conn.close()
        return donors

def clear_all_donors():
    """Clear all donors from database"""
    with db_lock:
        conn = sqlite3.connect('donors.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('DELETE FROM donors')
        conn.commit()
        conn.close()

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

# Initialize database
init_db()

# Get or create session ID for this user
if 'session_id' not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# Check if this is the admin session
admin_session_id = get_admin_session()

if admin_session_id is None:
    # No admin yet - this user becomes admin
    set_admin_session(st.session_state.session_id)
    is_admin = True
else:
    # Check if current user is the admin
    is_admin = (st.session_state.session_id == admin_session_id)

# ==================== ADMIN DASHBOARD MODE ====================
if is_admin:
    st.title("📊 Live Donation Dashboard")
    st.caption("🔒 Admin View")
    
    # Auto-refresh toggle and Stop button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=True)
    with col3:
        stop_event = st.button("🛑 Stop Event", type="primary")
    
    # Handle stop event
    if stop_event:
        st.session_state.event_stopped = True
        st.rerun()
    
    # Check if event is stopped
    if st.session_state.get('event_stopped', False):
        st.title("📋 Final Donation Report")
        st.caption("Event has been stopped")
        
        donors = get_all_donors()
        
        if donors:
            # Calculate total
            total = 0
            for donor in donors:
                amount_str = donor[3]
                try:
                    total += float(amount_str) if amount_str and amount_str.replace('.', '').replace(',', '').isdigit() else 0
                except:
                    pass
            
            # Summary metrics
            st.markdown("### 📊 Final Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💰 Total Raised", f"${total:,.2f}")
            with col2:
                st.metric("🎁 Total Donations", len(donors))
            with col3:
                avg = total / len(donors) if len(donors) > 0 else 0
                st.metric("📈 Average Donation", f"${avg:,.2f}")
            
            st.markdown("---")
            st.markdown("### 📋 All Donations")
            
            # Prepare data for table
            table_data = []
            for idx, donor in enumerate(donors, 1):
                name = donor[0] or "Anonymous"
                phone = donor[1] or "N/A"
                email = donor[2] or "N/A"
                amount = donor[3] or "0"
                timestamp = donor[4]
                
                # Parse timestamp
                try:
                    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    time_str = dt.strftime('%m/%d/%Y %I:%M %p')
                except:
                    time_str = timestamp
                
                table_data.append({
                    "#": idx,
                    "Name": name,
                    "Phone": phone,
                    "Email": email,
                    "Amount": f"${amount}",
                    "Time": time_str
                })
            
            df = pd.DataFrame(table_data)
            
            # Display table
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "#": st.column_config.NumberColumn("No.", width="small"),
                    "Name": st.column_config.TextColumn("Full Name", width="medium"),
                    "Phone": st.column_config.TextColumn("Phone", width="medium"),
                    "Email": st.column_config.TextColumn("Email", width="medium"),
                    "Amount": st.column_config.TextColumn("Amount", width="small"),
                    "Time": st.column_config.TextColumn("Timestamp", width="medium")
                }
            )
            
            # Download buttons
            st.markdown("---")
            st.markdown("### 📥 Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV download
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv,
                    file_name=f"donations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Excel download (requires openpyxl)
                try:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='Donations')
                    buffer.seek(0)
                    
                    st.download_button(
                        label="📊 Download Excel",
                        data=buffer,
                        file_name=f"donations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except ImportError:
                    st.info("Install openpyxl to enable Excel export")
            
            # Restart button
            st.markdown("---")
            if st.button("🔄 Start New Event"):
                clear_all_donors()
                st.session_state.event_stopped = False
                st.rerun()
        else:
            st.info("No donations recorded.")
            if st.button("🔄 Start New Event"):
                st.session_state.event_stopped = False
                st.rerun()
    
    else:
        # Normal dashboard view (when event is active)
        # Donation goal
        GOAL = 2000.0
        
        # Get all donors from database
        donors = get_all_donors()
        
        # Calculate total donations
        total = 0
        if donors:
            for donor in donors:
                amount_str = donor[3]  # amount is 4th column
                try:
                    total += float(amount_str) if amount_str and amount_str.replace('.', '').replace(',', '').isdigit() else 0
                except:
                    pass
        
        # Calculate progress percentage
        progress_percentage = (total / GOAL) * 100
        progress_percentage = min(progress_percentage, 100)  # Cap at 100%
        
        # Display goal and progress bar
        st.markdown(f"### 🎯 Fundraising Goal: ${GOAL:,.2f}")
        st.progress(progress_percentage / 100)
        st.markdown(f"**{progress_percentage:.1f}% Complete** — ${total:,.2f} of ${GOAL:,.2f} raised")
        
        if donors:
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💰 Total Raised", f"${total:,.2f}")
            with col2:
                st.metric("🎁 Total Donations", len(donors))
            with col3:
                remaining = max(GOAL - total, 0)
                st.metric("🎯 Remaining", f"${remaining:,.2f}")
            
            # Celebration when goal is reached
            if total >= GOAL:
                st.success("🎉 **GOAL REACHED!** Thank you to all our donors!")

            # Display donors
            st.markdown("---")
            st.markdown("## 🙌 Live Donor Wall")

            
            
            # Create a placeholder for the live donor card
            placeholder = st.empty()
            
            # Retrieve donors from DB
            donors = get_all_donors()
            
            if donors:
                # Get the newest donor (last entry)
                latest_donor = donors[0]  # because we order by timestamp DESC
                latest_id = latest_donor[0] if len(latest_donor) > 0 else None
            
                # Track the last shown donor in session state
                if "last_shown_donor" not in st.session_state:
                    st.session_state.last_shown_donor = None
            
                # If this donor hasn’t been shown yet
                if latest_donor != st.session_state.last_shown_donor:
                    name = latest_donor[0] or "Anonymous"
                    phone = latest_donor[1] or ""
                    email = latest_donor[2] or ""
                    amount = latest_donor[3] or "0"
                    timestamp = latest_donor[4]
            
                    # Format timestamp
                    try:
                        dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                        time_str = dt.strftime('%I:%M %p')
                    except:
                        time_str = timestamp
            
                    # Show animated donor card
                    card_html = f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea, #764ba2 100%);
                        border-radius: 20px;
                        padding: 30px;
                        box-shadow: 0 6px 15px rgba(0,0,0,0.15);
                        text-align: center;
                        margin: 20px 0;
                        font-family: 'Helvetica', sans-serif;
                        animation: fadeIn 1s ease-in-out;
                    ">
                        <h2 style='color:white;font-size: 3.5em;text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>{name}</h2>
                        <h3 style='color:#FFD700;font-size: 4.5em;font-weight: bold;text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>Donated ${amount}</h3>
                        <p style='color:yellow;'>Thank you for your generosity!</p>
                    </div>
                    <style>
                        @keyframes fadeIn {{
                            from {{ opacity: 0; transform: scale(0.9); }}
                            to {{ opacity: 1; transform: scale(1); }}
                        }}
                    </style>
                    """
            
                    placeholder.markdown(card_html, unsafe_allow_html=True)
                    st.balloons()
                    st.toast(f"💖 New donation from {name}: ${amount}")
            
                    # Keep visible for 5 seconds
                    time.sleep(5)
                    placeholder.empty()
            
                    # Remember we already showed this donor
                    st.session_state.last_shown_donor = latest_donor
            else:
                st.info("No donations yet. Waiting for uploads...")
            
            # Clear button
            st.markdown("---")
            if st.button("🗑️ Clear All Donors"):
                clear_all_donors()
                st.rerun()
        else:
            st.info("No donations yet. Waiting for uploads...")
        
        # Auto-refresh every 5 seconds if enabled
        if auto_refresh:
            import time
            time.sleep(5)
            st.rerun()

# ==================== UPLOAD STATION MODE ====================
else:
    st.title("📸 Donation Form Upload")
    st.write("Take a photo of the donation form to submit.")
    
    # Capture image from camera
    photo = st.camera_input("Take a photo of the donation form")
    
    if photo:
        # Display the photo
        image = Image.open(photo)
        st.image(image, caption="Captured Image", use_container_width=True)
        
        # Convert to bytes
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        
        # Extract form data
        with st.spinner("Analyzing photo..."):
            result = extract_form_data(image_bytes)
            
            if result:
                # Add to database
                add_donor(result)
                st.success("✅ Donation submitted successfully!")
                st.balloons()
                
                # Show extracted info
                st.markdown("### Extracted Information:")
                st.markdown(f"**Name:** {result['Full Name'] or 'Anonymous'}")
                st.markdown(f"**Amount:** ${result['Amount'] or '0'}")
                
                st.info("📤 This donation has been sent to the main dashboard.")
            else:
                st.error("Could not extract donor details. Please try again.")
