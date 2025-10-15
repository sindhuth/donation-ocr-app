import streamlit as st
from phi.agent import Agent
from phi.model.openai import OpenAIChat
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

# Initialize OpenAI models for agents
vision_model = OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
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

# Agent 1: Vision Agent - Analyzes photo and extracts data
vision_agent = Agent(
    name="Vision Analyzer",
    model=vision_model,
    description="Analyzes donation form images and extracts donor information",
    instructions=[
        "You are an expert at reading handwritten forms",
        "Extract the donor's full name and donation amount from images",
        "Try to match names with common South Indian names if applicable",
        "Analyze cursive and capital letters carefully",
        "Return only the name and amount in a structured format"
    ],
    markdown=True
)

# Agent 2: Data Processing Agent - Validates and processes extracted data
processing_agent = Agent(
    name="Data Processor",
    model=processing_model,
    description="Processes and validates extracted donor information",
    instructions=[
        "You validate and structure donor information",
        "Ensure name is properly formatted (title case)",
        "Ensure amount is a valid number",
        "Clean up any formatting issues"
    ],
    markdown=True
)

def encode_image(image_bytes):
    """Encode image to base64"""
    return base64.b64encode(image_bytes).decode('utf-8')

def extract_with_vision_agent(image_bytes):
    """Use Vision Agent to extract form data"""
    try:
        base64_image = encode_image(image_bytes)
        
        # Vision agent analyzes the image
        prompt = f"""Analyze this donation form image and extract:
1. Full Name of the donor
2. Donation Amount (number only, no currency symbols)

Return ONLY in this exact format:
Name: [extracted name]
Amount: [extracted number]

Image: data:image/jpeg;base64,{base64_image}
"""
        
        response = vision_agent.run(prompt)
        
        # Parse the response
        content = response.content if hasattr(response, 'content') else str(response)
        
        data = {'name': '', 'amount': ''}
        for line in content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if 'name' in key:
                    data['name'] = value
                elif 'amount' in key:
                    # Extract just the number
                    data['amount'] = ''.join(filter(lambda x: x.isdigit() or x == '.', value))
        
        return data
    except Exception as e:
        st.error(f"Error extracting data: {e}")
        return None

def process_with_data_agent(extracted_data):
    """Use Processing Agent to validate and clean data"""
    try:
        prompt = f"""Process this donor information:
Name: {extracted_data.get('name', '')}
Amount: {extracted_data.get('amount', '')}

Please:
1. Format the name properly (title case)
2. Ensure amount is a valid number
3. Return in the exact format:
Name: [cleaned name]
Amount: [cleaned amount]
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
                    data['name'] = value
                elif 'amount' in key:
                    data['amount'] = ''.join(filter(lambda x: x.isdigit() or x == '.', value))
        
        return data
    except Exception as e:
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
    
    # Auto-refresh toggle
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=True)
    with col3:
        if st.button("🗑️ Clear All"):
            clear_all_donors()
            st.rerun()
    
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
    
    # Progress bar
    progress_percentage = min((total / GOAL) * 100, 100)
    st.markdown(f"### 🎯 Fundraising Goal: ${GOAL:,.2f}")
    st.progress(progress_percentage / 100)
    st.markdown(f"**{progress_percentage:.1f}% Complete** — ${total:,.2f} of ${GOAL:,.2f} raised")
    
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
        
        # Donor list
        st.markdown("---")
        st.markdown("## 🙌 Recent Donations")
        
        for donor in donors[:10]:  # Show last 10
            name = donor[1] or "Anonymous"
            amount = donor[2] or "0"
            timestamp = donor[3]
            
            try:
                dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                time_str = dt.strftime('%I:%M %p')
            except:
                time_str = timestamp
            
            st.markdown(f"**{name}** donated **${amount}** at {time_str}")
    else:
        st.info("No confirmed donations yet.")
    
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
                st.balloons()
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
        with st.spinner("🤖 Agent 1: Analyzing photo..."):
            extracted = extract_with_vision_agent(image_bytes)
        
        if extracted:
            with st.spinner("🤖 Agent 2: Processing data..."):
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
                
                time.sleep(2)
                st.rerun()
        else:
            st.error("Could not extract information. Please try again.")
