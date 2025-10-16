import streamlit as st
from phi.agent import Agent
from phi.model.openai import OpenAIChat
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
import uuid
import re

# Initialize OpenAI client for vision
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize phidata agent for data processing
processing_model = OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# Database lock for thread-safe operations
db_lock = threading.Lock()

# ==================== DATABASE FUNCTIONS ====================
def init_db():
    """Initialize SQLite database"""
    with db_lock:
        conn = sqlite3.connect('donors.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS donors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT,
                amount TEXT,
                image_data BLOB,
                status TEXT DEFAULT 'pending',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS roles (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                admin_session TEXT,
                editor_session TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

def get_roles():
    """Get admin and editor session IDs"""
    with db_lock:
        conn = sqlite3.connect('donors.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('SELECT admin_session, editor_session FROM roles WHERE id = 1')
        result = c.fetchone()
        conn.close()
        return result if result else (None, None)

def set_role(role_type, session_id):
    """Set admin or editor session"""
    with db_lock:
        conn = sqlite3.connect('donors.db', check_same_thread=False)
        c = conn.cursor()
        
        # Get current roles
        c.execute('SELECT admin_session, editor_session FROM roles WHERE id = 1')
        result = c.fetchone()
        
        if result:
            admin, editor = result
            if role_type == 'admin':
                c.execute('UPDATE roles SET admin_session = ? WHERE id = 1', (session_id,))
            else:
                c.execute('UPDATE roles SET editor_session = ? WHERE id = 1', (session_id,))
        else:
            if role_type == 'admin':
                c.execute('INSERT INTO roles (id, admin_session) VALUES (1, ?)', (session_id,))
            else:
                c.execute('INSERT INTO roles (id, editor_session) VALUES (1, ?)', (session_id,))
        
        conn.commit()
        conn.close()

def add_donor_pending(image_bytes, extracted_data):
    """Add donor with pending status"""
    with db_lock:
        conn = sqlite3.connect('donors.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('''
            INSERT INTO donors (full_name, amount, image_data, status)
            VALUES (?, ?, ?, 'pending')
        ''', (extracted_data.get('name', ''), extracted_data.get('amount', ''), image_bytes))
        donor_id = c.lastrowid
        conn.commit()
        conn.close()
        return donor_id

def get_pending_donors():
    """Get all pending donors"""
    with db_lock:
        conn = sqlite3.connect('donors.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('SELECT id, full_name, amount, image_data, timestamp FROM donors WHERE status = "pending" ORDER BY timestamp ASC')
        donors = c.fetchall()
        conn.close()
        return donors

def update_donor(donor_id, name, amount):
    """Update donor and mark as confirmed"""
    with db_lock:
        conn = sqlite3.connect('donors.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('''
            UPDATE donors 
            SET full_name = ?, amount = ?, status = 'confirmed'
            WHERE id = ?
        ''', (name, amount, donor_id))
        conn.commit()
        conn.close()

def get_confirmed_donors():
    """Get all confirmed donors"""
    with db_lock:
        conn = sqlite3.connect('donors.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('SELECT id, full_name, amount, timestamp FROM donors WHERE status = "confirmed" ORDER BY timestamp DESC')
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

# ==================== PHIDATA AGENTS ====================

def encode_image(image_bytes):
    """Encode image to base64"""
    return base64.b64encode(image_bytes).decode('utf-8')

# Agent 1: Vision Analysis using OpenAI's native vision API
def extract_with_vision_agent(image_bytes):
    """Use OpenAI Vision API to extract form data from image"""
    try:
        base64_image = encode_image(image_bytes)
        
        # Use OpenAI's vision API directly
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract handwritten donor details from this donation form:\n"
                                "1. Full Name of the donor\n"
                                "2. Donation Amount (number only, no currency symbols)\n\n"
                                "Important:\n"
                                "- Try to match names with common South Indian names if applicable\n"
                                "- Analyze cursive and capital letters carefully\n"
                                "- Look for fields labeled 'Name', 'Donor', 'Full Name', etc.\n"
                                "- Look for fields labeled 'Amount', 'Donation', 'Rs', '$', etc.\n\n"
                                "Return ONLY in this exact format:\n"
                                "Name: [extracted name]\n"
                                "Amount: [extracted number only]"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse the response
        data = {'name': '', 'amount': ''}
        for line in content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if 'name' in key:
                    data['name'] = value
                elif 'amount' in key:
                    # Extract just numbers and decimal point
                    numbers = re.findall(r'\d+\.?\d*', value)
                    if numbers:
                        data['amount'] = numbers[0]
        
        return data
    except Exception as e:
        st.error(f"Error extracting data from image: {e}")
        return None

# Agent 2: Data Processing Agent - Validates and processes extracted data
processing_agent = Agent(
    name="Data Processor",
    model=processing_model,
    description="Processes and validates extracted donor information",
    instructions=[
        "You validate and structure donor information",
        "Ensure name is properly formatted (title case, fix spacing)",
        "Ensure amount is a valid number",
        "Clean up any formatting issues",
        "Fix common OCR errors in names"
    ],
    markdown=True
)

def process_with_data_agent(extracted_data):
    """Use Processing Agent to validate and clean data"""
    try:
        prompt = f"""Process and validate this donor information:
Name: {extracted_data.get('name', '')}
Amount: {extracted_data.get('amount', '')}

Please:
1. Format the name properly (title case, fix spacing)
2. Ensure amount is a valid number (remove any non-numeric characters except decimal point)
3. Fix common OCR errors
4. If name is empty or unclear, return "Anonymous"
5. If amount is empty or invalid, return "0"

Return in this exact format:
Name: [cleaned name]
Amount: [cleaned number only]
"""
        
        response = processing_agent.run(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        data = {'name': '', 'amount': ''}
        for line in content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if 'name' in key:
                    data['name'] = value if value else 'Anonymous'
                elif 'amount' in key:
                    # Extract just numbers and decimal
                    numbers = re.findall(r'\d+\.?\d*', value)
                    if numbers:
                        data['amount'] = numbers[0]
                    else:
                        data['amount'] = '0'
        
        # Fallback if parsing failed
        if not data['name']:
            data['name'] = extracted_data.get('name', 'Anonymous')
        if not data['amount']:
            data['amount'] = extracted_data.get('amount', '0')
            
        return data
    except Exception as e:
        st.warning(f"Data processing warning: {e}")
        return extracted_data

# ==================== INITIALIZE ====================
init_db()

# Get or create session ID
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Determine user role
admin_session, editor_session = get_roles()

if admin_session is None:
    # First user becomes admin
    set_role('admin', st.session_state.session_id)
    user_role = 'admin'
elif editor_session is None and st.session_state.session_id != admin_session:
    # Second user becomes editor
    set_role('editor', st.session_state.session_id)
    user_role = 'editor'
elif st.session_state.session_id == admin_session:
    user_role = 'admin'
elif st.session_state.session_id == editor_session:
    user_role = 'editor'
else:
    user_role = 'donor'

# ==================== ADMIN SCREEN ====================
if user_role == 'admin':
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
        
        donors = get_confirmed_donors()
        
        if donors:
            # Calculate total
            total = 0
            for donor in donors:
                amount_str = donor[2]
                try:
                    total += float(amount_str) if amount_str else 0
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
                donor_id = donor[0]
                name = donor[1] or "Anonymous"
                amount = donor[2] or "0"
                timestamp = donor[3]
                
                # Parse timestamp
                try:
                    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    time_str = dt.strftime('%m/%d/%Y %I:%M %p')
                except:
                    time_str = timestamp
                
                table_data.append({
                    "#": idx,
                    "Name": name,
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
                # Excel download
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
        
        # Get confirmed donors
        donors = get_confirmed_donors()
        
        # Calculate total
        total = 0
        if donors:
            for donor in donors:
                amount_str = donor[2]
                try:
                    total += float(amount_str) if amount_str else 0
                except:
                    pass
        
        # Progress bar and pot visualization
        progress_percentage = min((total / GOAL) * 100, 100)
        st.markdown(f"### 🎯 Fundraising Goal: ${GOAL:,.2f}")
        
        # Create two columns for progress bar and pot
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.progress(progress_percentage / 100)
            st.markdown(f"**{progress_percentage:.1f}% Complete** — ${total:,.2f} of ${GOAL:,.2f} raised")
        
        with col_right:
            # Realistic pot filling with gold coins
            fill_height = int(progress_percentage * 2)  # Scale to 200px max height
            num_coins = int(progress_percentage / 10)  # More coins as we progress
            
            pot_html = f"""
            <div style="display: flex; justify-content: center; align-items: flex-end; height: 250px; position: relative;">
                <!-- Gold coins filling up -->
                <div style="
                    position: absolute;
                    bottom: 50px;
                    width: 180px;
                    height: {fill_height}px;
                    background: linear-gradient(180deg, 
                        #FFD700 0%, 
                        #FFA500 20%,
                        #FFD700 40%,
                        #DAA520 60%,
                        #FFD700 80%,
                        #FFA500 100%);
                    border-radius: 0 0 80px 80px;
                    box-shadow: inset 0 -10px 20px rgba(0,0,0,0.2),
                                inset 0 10px 20px rgba(255,255,255,0.3);
                    overflow: hidden;
                    transition: height 0.5s ease;
                ">
                    <!-- Animated coin sparkles -->
                    <div style="
                        width: 100%;
                        height: 100%;
                        background-image: 
                            radial-gradient(circle at 20% 30%, rgba(255,255,255,0.4) 2px, transparent 2px),
                            radial-gradient(circle at 70% 60%, rgba(255,255,255,0.3) 3px, transparent 3px),
                            radial-gradient(circle at 40% 80%, rgba(255,255,255,0.4) 2px, transparent 2px),
                            radial-gradient(circle at 80% 20%, rgba(255,255,255,0.3) 2px, transparent 2px);
                        background-size: 50px 50px;
                        animation: shimmer 3s linear infinite;
                    "></div>
                </div>
                
                <!-- Realistic pot -->
                <div style="
                    width: 200px;
                    height: 200px;
                    background: linear-gradient(180deg, 
                        #8B4513 0%, 
                        #654321 20%,
                        #8B4513 40%,
                        #A0522D 60%,
                        #654321 100%);
                    border-radius: 0 0 90px 90px;
                    position: relative;
                    box-shadow: 
                        inset -20px 0 40px rgba(0,0,0,0.4),
                        inset 20px 0 40px rgba(139,69,19,0.3),
                        0 20px 40px rgba(0,0,0,0.5);
                    border: 3px solid #654321;
                ">
                    <!-- Pot rim -->
                    <div style="
                        position: absolute;
                        top: -15px;
                        left: -10px;
                        width: 220px;
                        height: 30px;
                        background: linear-gradient(180deg, #A0522D 0%, #654321 100%);
                        border-radius: 15px;
                        box-shadow: 
                            0 5px 10px rgba(0,0,0,0.5),
                            inset 0 -5px 10px rgba(0,0,0,0.3);
                    "></div>
                    
                    <!-- Pot handles -->
                    <div style="
                        position: absolute;
                        left: -25px;
                        top: 30px;
                        width: 30px;
                        height: 60px;
                        border: 8px solid #654321;
                        border-right: none;
                        border-radius: 40px 0 0 40px;
                        box-shadow: -5px 5px 10px rgba(0,0,0,0.4);
                    "></div>
                    <div style="
                        position: absolute;
                        right: -25px;
                        top: 30px;
                        width: 30px;
                        height: 60px;
                        border: 8px solid #654321;
                        border-left: none;
                        border-radius: 0 40px 40px 0;
                        box-shadow: 5px 5px 10px rgba(0,0,0,0.4);
                    "></div>
                    
                    <!-- Inner shadow for depth -->
                    <div style="
                        position: absolute;
                        top: 15px;
                        left: 10px;
                        right: 10px;
                        bottom: 10px;
                        border-radius: 0 0 80px 80px;
                        box-shadow: inset 0 20px 40px rgba(0,0,0,0.6);
                    "></div>
                </div>
                
                <!-- Overflow coins when full -->
                {f'''
                <div style="
                    position: absolute;
                    top: -20px;
                    display: flex;
                    gap: 10px;
                    animation: coinPop 0.5s ease;
                ">
                    <div style="
                        width: 30px;
                        height: 30px;
                        background: radial-gradient(circle at 30% 30%, #FFD700, #DAA520);
                        border-radius: 50%;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.3), inset -2px -2px 4px rgba(0,0,0,0.2);
                    "></div>
                    <div style="
                        width: 25px;
                        height: 25px;
                        background: radial-gradient(circle at 30% 30%, #FFD700, #DAA520);
                        border-radius: 50%;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.3), inset -2px -2px 4px rgba(0,0,0,0.2);
                    "></div>
                </div>
                ''' if progress_percentage >= 100 else ''}
            </div>
            
            <style>
                @keyframes shimmer {{
                    0% {{ transform: translateY(0); }}
                    100% {{ transform: translateY(-50px); }}
                }}
                @keyframes coinPop {{
                    0% {{ transform: translateY(20px) scale(0); opacity: 0; }}
                    50% {{ transform: translateY(-10px) scale(1.2); opacity: 1; }}
                    100% {{ transform: translateY(0) scale(1); opacity: 1; }}
                }}
            </style>
            """
            
            st.markdown(pot_html, unsafe_allow_html=True)
        
        if donors:
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💰 Total Raised", f"${total:,.2f}")
            with col2:
                st.metric("🎁 Total Donations", len(donors))
            with col3:
                remaining = max(GOAL - total, 0)
                st.metric("🎯 Remaining", f"${remaining:,.2f}")
            
            if total >= GOAL:
                st.success("🎉 **GOAL REACHED!** Thank you to all our donors!")
            
            # Display latest donor animation
            st.markdown("---")
            st.markdown("## 🙌 Live Donor Wall")
            
            # Create a placeholder for the live donor card
            placeholder = st.empty()
            
            # Get the newest donor
            latest_donor = donors[0]  # First one (ordered DESC by timestamp)
            
            # Track the last shown donor in session state
            if "last_shown_donor_id" not in st.session_state:
                st.session_state.last_shown_donor_id = None
            
            latest_donor_id = latest_donor[0]
            
            # If this donor hasn't been shown yet
            if latest_donor_id != st.session_state.last_shown_donor_id:
                name = latest_donor[1] or "Anonymous"
                amount = latest_donor[2] or "0"
                timestamp = latest_donor[3]
                
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
                    <h3 style='color:#FFD700;font-size: 4.5em;font-weight: bold;text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>${amount}</h3>
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
                
                # Keep visible for 3 seconds
                time.sleep(3)
                placeholder.empty()
                
                # Remember we already showed this donor
                st.session_state.last_shown_donor_id = latest_donor_id
            
            # Clear button
            st.markdown("---")
            if st.button("🗑️ Clear All Donors"):
                clear_all_donors()
                st.session_state.last_shown_donor_id = None
                st.rerun()
        else:
            st.info("No confirmed donations yet. Waiting for editor approval...")
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(5)
            st.rerun()

# ==================== EDITOR SCREEN ====================
elif user_role == 'editor':
    st.title("✏️ Donation Editor")
    st.caption("🔐 Editor View - Review and confirm donations")
    
    # Get pending donors
    pending = get_pending_donors()
    
    if pending:
        st.markdown(f"### 📋 Pending Donations: {len(pending)}")
        
        # Process first pending donor
        donor = pending[0]
        donor_id, name, amount, image_data, timestamp = donor
        
        # Display image
        st.markdown("#### 📸 Uploaded Form")
        image = Image.open(io.BytesIO(image_data))
        st.image(image, caption="Donation Form", use_container_width=True)
        
        # Editable fields
        st.markdown("#### ✏️ Edit Details")
        
        col1, col2 = st.columns(2)
        with col1:
            edited_name = st.text_input("Full Name", value=name, key=f"name_{donor_id}")
        with col2:
            edited_amount = st.text_input("Amount", value=amount, key=f"amount_{donor_id}")
        
        # Confirm button
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✅ Confirm Donation", type="primary", use_container_width=True):
                update_donor(donor_id, edited_name, edited_amount)
                st.success(f"✅ Donation confirmed for {edited_name}!")
                time.sleep(1)
                st.rerun()
        
        with col2:
            if st.button("❌ Skip", use_container_width=True):
                # Mark as confirmed but with a flag or delete
                update_donor(donor_id, edited_name, edited_amount)
                st.rerun()
        
        # Show queue
        if len(pending) > 1:
            st.markdown(f"**{len(pending) - 1} more in queue**")
    else:
        st.info("🎉 No pending donations. Waiting for uploads...")
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Auto-refresh every 3 seconds
    time.sleep(3)
    st.rerun()

# ==================== DONOR UPLOAD SCREEN ====================
else:
    st.title("📸 Donation Form Upload")
    st.write("Take a photo of the donation form to submit.")
    
    # Camera input
    photo = st.camera_input("Take a photo of the donation form")
    
    if photo:
        # Display the photo
        image = Image.open(photo)
        st.image(image, caption="Captured Image", use_container_width=True)
        
        # Convert to bytes
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        
        # Extract with agents
        with st.spinner("🤖 Agent 1: Analyzing photo with Vision AI..."):
            extracted = extract_with_vision_agent(image_bytes)
            
            if extracted:
                st.success(f"✅ Detected - Name: {extracted.get('name', 'N/A')}, Amount: ${extracted.get('amount', '0')}")
        
        if extracted:
            with st.spinner("🤖 Agent 2: Processing and validating data..."):
                processed = process_with_data_agent(extracted)
            
            if processed:
                # Add to database as pending
                donor_id = add_donor_pending(image_bytes, processed)
                
                st.success("✅ Form uploaded successfully!")
                st.balloons()
                
                # Show extracted info
                st.markdown("### 📋 Extracted Information:")
                st.markdown(f"**Name:** {processed['name'] or 'Not detected'}")
                st.markdown(f"**Amount:** ${processed['amount'] or '0'}")
                
                st.info("📤 Sent to editor for review and confirmation.")
                
                # Don't auto-rerun, let user take another photo if needed
        else:
            st.error("Could not extract information. Please try again.")
