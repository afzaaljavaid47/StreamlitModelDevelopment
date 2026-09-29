import streamlit as st
import requests
import json
import os
import sqlite3
import time
import uuid
from contextlib import closing
from datetime import datetime
from streamlit_local_storage import LocalStorage
from streamlit_chat import message

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AI Learners Chat Bot Assistant",
    page_icon="🤖",
    layout="wide"
)

# ============================================================
# CONFIGURATION
# ============================================================
API_KEYS_FILE = "api_keys.json"
DB_FILE = "chat_history.db"
STORAGE_KEY = "chat_messages"
STATUS_LABELS = {
    "active": "Ended by user",
    "ticket": "Ticket created",
    "live_agent": "Live agent requested"
}
GREETING = [
    {
        "role": "assistant",
        "content": (
            "Hello! 👋 I'm your AI support assistant.\n\n"
            "How can I help you today?"
        )
    }
]

# ============================================================
# API KEY FUNCTIONS
# ============================================================
def load_api_keys():

    if not os.path.exists(API_KEYS_FILE):
        return []

    try:
        with open(API_KEYS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        return []

    except (json.JSONDecodeError, OSError):
        return []
def save_api_keys(api_keys):
    with open(API_KEYS_FILE, "w", encoding="utf-8") as file:
        json.dump(api_keys, file, indent=4)
def mask_api_key(api_key):
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return (
        api_key[:4]
        + "*" * (len(api_key) - 8)
        + api_key[-4:]
    )
def get_default_api_key():
    for key in load_api_keys():
        if key.get("is_default", False):
            return key
    return None
def add_api_key(name, provider, api_key):
    api_keys = load_api_keys()
    is_first_key = len(api_keys) == 0
    api_keys.append({
        "id": str(uuid.uuid4()),
        "name": name,
        "provider": provider,
        "api_key": api_key,
        "is_default": is_first_key
    })
    save_api_keys(api_keys)
def set_default_api_key(key_id):
    api_keys = load_api_keys()
    for key in api_keys:
        key["is_default"] = (key.get("id") == key_id)
    save_api_keys(api_keys)
def delete_api_key(key_id):
    api_keys = load_api_keys()
    deleted_key = None
    remaining_keys = []
    for key in api_keys:
        if key.get("id") == key_id:
            deleted_key = key
        else:
            remaining_keys.append(key)
    if deleted_key and deleted_key.get("is_default"):
        if remaining_keys:
            remaining_keys[0]["is_default"] = True
    save_api_keys(remaining_keys)

# ============================================================
# CHAT HISTORY DATABASE (SQLite)
# ============================================================
def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
def init_db():
    with closing(get_db()) as conn:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    status TEXT,
                    intent TEXT,
                    urgency TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    position INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    FOREIGN KEY (chat_id)
                        REFERENCES chats (id)
                        ON DELETE CASCADE
                )
                """
            )
def default_chat_name(messages):
    for message in messages:
        if message.get("role") == "user":
            text = " ".join(
                str(message.get("content", "")).split()
            )
            if len(text) > 40:
                text = text[:40].rstrip() + "…"
            return text or "Untitled chat"

    return "Untitled chat"
def save_chat_to_db(    name,    messages,    status,    intent=None,    urgency=None):
    with closing(get_db()) as conn:
        with conn:

            cursor = conn.execute(
                """
                INSERT INTO chats
                    (name, status, intent, urgency, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    name,
                    status,
                    intent,
                    urgency,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
            )

            chat_id = cursor.lastrowid

            conn.executemany(
                """
                INSERT INTO chat_messages
                    (chat_id, position, role, content)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        chat_id,
                        position,
                        message["role"],
                        message["content"]
                    )
                    for position, message in enumerate(messages)
                ]
            )

    return chat_id
def list_chats(search=""):
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT
                c.id,
                c.name,
                c.status,
                c.intent,
                c.urgency,
                c.created_at,
                COUNT(m.id) AS message_count
            FROM chats c
            LEFT JOIN chat_messages m ON m.chat_id = c.id
            WHERE c.name LIKE ?
            GROUP BY c.id
            ORDER BY c.id DESC
            """,
            (f"%{search}%",)
        ).fetchall()

    return [dict(row) for row in rows]
def get_chat_messages(chat_id):
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT role, content
            FROM chat_messages
            WHERE chat_id = ?
            ORDER BY position ASC
            """,
            (chat_id,)
        ).fetchall()
    return [dict(row) for row in rows]
def rename_chat(chat_id, new_name):
    with closing(get_db()) as conn:
        with conn:
            conn.execute(
                "UPDATE chats SET name = ? WHERE id = ?",
                (new_name, chat_id)
            )
def delete_chat(chat_id):
    with closing(get_db()) as conn:
        with conn:
            conn.execute(
                "DELETE FROM chats WHERE id = ?",
                (chat_id,)
            )

# Create tables on startup (safe to call on every run)
init_db()

# ============================================================
# JEV API
# ============================================================
def CallJevAPI(conversation):
    default_key = get_default_api_key()
    if not default_key:
        return {
            "success": False,
            "error": "No default JEV API key configured."
        }
    api_key = default_key.get("api_key")
    if not api_key:
        return {
            "success": False,
            "error": "JEV API key is empty."
        }
    url = "https://api.typesafe.ai/v1/systemone"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # --------------------------------------------------------
    # Convert conversation into compact state
    # --------------------------------------------------------
    conversation_text = ""
    for message in conversation:
        role = message["role"]
        content = message["content"]
        conversation_text += f"{role.upper()}: {content}\n"
    # --------------------------------------------------------
    # JEV questions
    # --------------------------------------------------------
    questions = {
        "intent": {
            "type": "choice",
            "instructions": (
                "Identify the main customer support intent."
            ),
            "criteria": {
                "billing": (
                    "Payment, charge, refund, invoice, "
                    "subscription billing or pricing."
                ),
                "technical": (
                    "Technical problem, bug, error, "
                    "API problem or system failure."
                ),
                "account": (
                    "Login, account, profile, password, "
                    "access or account settings."
                ),
                "order": (
                    "Order, delivery, shipping, "
                    "purchase or order status."
                ),
                "general": (
                    "General question that does not "
                    "fit another category."
                )
            }
        },

        "urgency": {
            "type": "score",
            "instructions": "How urgent is this customer issue?",
            "criteria": ["Low", "Medium", "High", "Critical"]
        },

        "needs_human": {
            "type": "noul",
            "instructions": (
                "Does this conversation require "
                "human support?"
            )
        },

        "needs_more_information": {
            "type": "noul",
            "instructions": (
                "Is important information still missing "
                "before this issue can be resolved or "
                "correctly routed?"
            )
        },

        "create_ticket": {
            "type": "noul",
            "instructions": (
                "Should this issue be converted into "
                "a support ticket instead of continuing "
                "the automated conversation?"
            )
        },

        "live_agent": {
            "type": "noul",
            "instructions": (
                "Should this customer be transferred "
                "to a live support agent now?"
            )
        },

        "next_question": {
            "type": "choice",
            "instructions": (
                "Choose the most useful next question "
                "category when more information is needed."
            ),
            "criteria": {
                "issue_details": (
                    "Ask what exactly happened and "
                    "what problem the customer is experiencing."
                ),
                "order_id": (
                    "Ask for the order ID or transaction ID."
                ),
                "error_message": (
                    "Ask for the exact error message or error code."
                ),
                "account_details": (
                    "Ask for the relevant account information."
                ),
                "payment_details": (
                    "Ask for relevant payment or transaction details."
                ),
                "none": (
                    "No additional question is required."
                )
            }
        }
    }
    payload = {
        "model": "jev-latest",
        "state": {
            "conversation": conversation_text
        },
        "questions": questions
    }
    response = None
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        return {
            "success": True,
            "data": data
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "JEV API request timed out."
        }

    except requests.exceptions.HTTPError:
        return {
            "success": False,
            "error": (
                f"JEV API HTTP error: "
                f"{response.status_code if response is not None else 'unknown'}"
            ),
            "response": response.text if response is not None else ""
        }

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"JEV API request failed: {str(e)}"
        }

    except ValueError:
        return {
            "success": False,
            "error": "JEV returned invalid JSON."
        }

# ============================================================
# PARSE JEV DECISION
# ============================================================
def parse_jev_decision(result):
    if not result.get("success"):
        return None
    try:
        answers = result["data"]["answers"]
        # ----------------------------------------------------
        # Intent
        # ----------------------------------------------------
        intent = answers["intent"]["choice"]
        # ----------------------------------------------------
        # Urgency
        #
        # JEV score starts at 0:
        # 0 = Low, 1 = Medium, 2 = High, 3 = Critical
        # ----------------------------------------------------
        urgency_score = round(answers["urgency"]["score"])
        urgency_levels = {
            0: "Low",
            1: "Medium",
            2: "High",
            3: "Critical"
        }
        urgency = urgency_levels.get(urgency_score, "Unknown")
        # ----------------------------------------------------
        # Noul values -> percentages
        # ----------------------------------------------------
        needs_human = round(answers["needs_human"]["noul"] * 100)
        needs_more_information = round(
            answers["needs_more_information"]["noul"] * 100
        )
        create_ticket = round(answers["create_ticket"]["noul"] * 100)
        live_agent = round(answers["live_agent"]["noul"] * 100)
    # ----------------------------------------------------
        # Next question
        # ----------------------------------------------------
        next_question = answers["next_question"]["choice"]
        return {
            "intent": intent,
            "urgency": urgency,
            "urgency_score": urgency_score,
            "needs_human": needs_human,
            "needs_more_information": needs_more_information,
            "create_ticket": create_ticket,
            "live_agent": live_agent,
            "next_question": next_question
        }

    except (KeyError, TypeError, ValueError) as e:
        return {"parse_error": str(e)}

# ============================================================
# QUESTION GENERATOR
# ============================================================
def get_question(next_question):
    questions = {
        "issue_details": (
            "Could you please explain exactly "
            "what happened?"
        ),
        "order_id": (
            "Could you please provide your "
            "order or transaction ID?"
        ),
        "error_message": (
            "What exact error message or "
            "error code are you seeing?"
        ),
        "account_details": (
            "Could you please provide the relevant "
            "account information?"
        ),

        "payment_details": (
            "Could you please provide the relevant "
            "payment or transaction details?"
        ),

        "none": (
            "Thank you. I have enough information "
            "to process your request."
        )
    }

    return questions.get(
        next_question,
        questions["issue_details"]
    )

# ============================================================
# ASSISTANT RESPONSE
# ============================================================
def generate_assistant_reply(decision):
    if "parse_error" in decision:
        return (
            "I'm sorry, I couldn't process "
            "your request correctly."
        )
    # Live agent
    if decision["live_agent"] >= 75:
        return (
            "Thank you for providing that information. "
            "Based on your request, I think this "
            "would be better handled by one of our "
            "live support agents.\n\n"
            "I'm preparing your conversation for "
            "live support now."
        )
    # Ticket
    if decision["create_ticket"] >= 75:
        return (
            "Thank you. I have collected the information "
            "needed for your request.\n\n"
            "I'm preparing a support ticket so our "
            "support team can continue working on it."
        )
    # Need more information
    if decision["needs_more_information"] >= 60:
        return get_question(decision["next_question"])
    # Human support
    if decision["needs_human"] >= 75:
        return (
            "I understand. This request requires "
            "additional support from our team.\n\n"
            "Would you like me to connect you "
            "with a live support agent?"
        )
    # Continue conversation
    return (
        "Thank you for the information. "
        "I can continue helping you with this issue. "
        "Could you tell me a little more about "
        "what you need?"
    )

# ============================================================
# CREATE TICKET
# ============================================================
def create_support_ticket(conversation, decision):
    ticket_id = "TKT-" + uuid.uuid4().hex[:8].upper()
    ticket = {
        "ticket_id": ticket_id,
        "intent": decision["intent"],
        "urgency": decision["urgency"],
        "conversation": conversation
    }
    # TODO: Replace this with your database/API call
    os.makedirs("tickets", exist_ok=True)
    with open( f"tickets/{ticket_id}.json", "w", encoding="utf-8") as file:
        json.dump(ticket, file, indent=4)
    return ticket_id

# ============================================================
# LIVE AGENT
# ============================================================
def request_live_agent(conversation, decision):
    session_id = "LIVE-" + uuid.uuid4().hex[:8].upper()
    live_request = {
        "session_id": session_id,
        "intent": decision["intent"],
        "urgency": decision["urgency"],
        "conversation": conversation
    }
    # TODO: Replace this with:
    # WebSocket / Redis / Database / CRM / Helpdesk / Live-chat provider
    os.makedirs("live_support", exist_ok=True)
    with open(
        f"live_support/{session_id}.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(live_request, file, indent=4)
    return session_id

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("⚙️ Settings")
page = st.sidebar.radio(
    "Choose Section",
    [
        "Live Chat Support",
        "Chat History",
        "APIs"
    ]
)

# ============================================================
# LIVE CHAT
# ============================================================
if page == "Live Chat Support":
    st.title("🤖 AI Learners Chat Bot Assistant")
    st.write("Welcome to the AI Learners live chat support.")
    st.divider()
    # --------------------------------------------------------
    # Local Storage
    # --------------------------------------------------------
    localS = LocalStorage()
    # --------------------------------------------------------
    # API status
    # --------------------------------------------------------
    default_key = get_default_api_key()
    if default_key:
        st.success(
            "✅ JEV API configured: "
            f"**{default_key.get('name', 'Unnamed Key')}**"
        )
    else:
        st.warning("⚠️ No JEV API key has been configured.")
        st.info("Go to **APIs** and add your JEV API key.")

    st.divider()

    # --------------------------------------------------------
    # Initialize session state
    # --------------------------------------------------------

    if "messages" not in st.session_state:
        st.session_state.messages = None

    if "chat_status" not in st.session_state:
        st.session_state.chat_status = "active"

    if "last_decision" not in st.session_state:
        st.session_state.last_decision = None

    if "end_chat_pending" not in st.session_state:
        st.session_state.end_chat_pending = False

    if "ending_chat" not in st.session_state:
        st.session_state.ending_chat = False

    if "saved_chat_name" not in st.session_state:
        st.session_state.saved_chat_name = None

    # --------------------------------------------------------
    # Handle "End Chat" (requested on the previous run)
    #
    # The button below only sets a flag and reruns. The actual
    # delete happens here, at the top of the next run, so the
    # localStorage component is NOT torn down by st.rerun().
    # --------------------------------------------------------

    if st.session_state.end_chat_pending:

        # The chat was already saved to the database by the
        # "Save & End Chat" button. Here we only clear the
        # browser storage and reset the session.

        localS.deleteItem(
            STORAGE_KEY,
            key="delete_chat_messages"
        )

        saved_name = st.session_state.saved_chat_name

        st.session_state.messages = list(GREETING)
        st.session_state.chat_status = "active"
        st.session_state.last_decision = None
        st.session_state.end_chat_pending = False
        st.session_state.saved_chat_name = None

        if saved_name:
            st.success(
                f"✅ Chat ended and saved as **{saved_name}**. "
                "You can view it in **Chat History**."
            )
        else:
            st.success("✅ Chat ended successfully.")

    # --------------------------------------------------------
    # Load conversation from localStorage
    # --------------------------------------------------------

    if st.session_state.messages is None:

        saved_chat = localS.getItem(STORAGE_KEY)

        # On the very first run the browser hasn't answered yet,
        # so getItem() returns None. Give it one more run before
        # falling back to the greeting.
        if (
            saved_chat is None
            and not st.session_state.get("storage_wait_done")
        ):
            st.session_state.storage_wait_done = True
            time.sleep(0.5)
            st.rerun()

        restored = None

        if isinstance(saved_chat, list):
            restored = saved_chat

        elif isinstance(saved_chat, str):
            try:
                parsed = json.loads(saved_chat)
                if isinstance(parsed, list):
                    restored = parsed
            except json.JSONDecodeError:
                restored = None

        st.session_state.messages = restored or list(GREETING)

    # --------------------------------------------------------
    # Handle input FIRST (before drawing messages)
    # --------------------------------------------------------

    user_message = None
    decision = None

    if st.session_state.chat_status == "active":
        user_message = st.chat_input("Type your message...")

    if user_message:

        # If the save prompt was open, close it and keep chatting
        st.session_state.ending_chat = False

        # Add user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_message
            }
        )

        # JEV decision
        with st.spinner("🤖 Analyzing your request..."):
            jev_result = CallJevAPI(st.session_state.messages)

        decision = parse_jev_decision(jev_result)

        # API failure
        if decision is None:

            assistant_response = (
                "I'm temporarily unable to "
                "analyze your request. "
                "Please try again."
            )

        elif "parse_error" in decision:

            assistant_response = (
                "I received an unexpected response "
                "from the decision service."
            )

        else:

            st.session_state.last_decision = decision

            if decision["live_agent"] >= 75:

                session_id = request_live_agent(
                    st.session_state.messages,
                    decision
                )

                st.session_state.chat_status = "live_agent"

                assistant_response = (
                    "Thank you. I've reviewed "
                    "your request and I'm connecting "
                    "you with a live support agent.\n\n"
                    f"Support session: **{session_id}**"
                )

            elif decision["create_ticket"] >= 75:

                ticket_id = create_support_ticket(
                    st.session_state.messages,
                    decision
                )

                st.session_state.chat_status = "ticket"

                assistant_response = (
                    "Thank you. I've collected the "
                    "information needed for your request.\n\n"
                    f"🎫 Your support ticket is "
                    f"**{ticket_id}**.\n\n"
                    "Our support team can now continue "
                    "working on your request."
                )

            else:

                assistant_response = generate_assistant_reply(decision)

        # Add assistant response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_response
            }
        )

        # Save active conversation to localStorage.
        # NOTE: no st.rerun() after this call, otherwise the
        # component that performs the write can be cancelled.
        localS.setItem(
            STORAGE_KEY,
            json.dumps(st.session_state.messages),
            key=f"save_chat_messages_{len(st.session_state.messages)}"
        )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if st.session_state.chat_status == "active" and any(
        m.get("role") == "user"
        for m in st.session_state.messages
    ):
        st.success("🟢 Chat Active")

    elif st.session_state.chat_status == "ticket":
        st.warning("🎫 Ticket Created")

    elif st.session_state.chat_status == "live_agent":
        st.info("👨‍💼 Waiting for Live Support")

    # --------------------------------------------------------
    # Display messages
    # --------------------------------------------------------
    for i,chat_message in enumerate(st.session_state.messages):
        message(
            chat_message["content"],
            is_user=(chat_message["role"] == "user"),
            key=f"live_chat_message_{i}"
        )
    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.write(message["content"])
    #         message(
    #             message["content"],
    #             is_user=(message["role"] == "user")
    #         )

    # --------------------------------------------------------
    # Show decision information (only for the message just sent)
    # --------------------------------------------------------

    # if (
    #     user_message
    #     and decision
    #     and "parse_error" not in decision
    # ):

        # with st.expander("🤖 JEV Decision", expanded=True):
        #
        #     col1, col2, col3 = st.columns(3)
        #
        #     with col1:
        #         st.metric("Intent", decision["intent"])
        #
        #     with col2:
        #         st.metric("Urgency", decision["urgency"])
        #
        #     with col3:
        #         st.metric("Human", f"{decision['needs_human']}%")

    # --------------------------------------------------------
    # End Chat
    # --------------------------------------------------------

    # Only show the button once the user has sent at least one message.
    # After End Chat the messages reset to the greeting, so the button
    # disappears until the user types again.

    has_user_message = any(
        m.get("role") == "user"
        for m in st.session_state.messages
    )

    if has_user_message:

        st.divider()

        if not st.session_state.ending_chat:

            if st.button("🔴 End Chat", use_container_width=True):

                st.session_state.ending_chat = True
                st.rerun()

        else:

            st.subheader("💾 Save this chat")

            chat_name = st.text_input(
                "Chat name",
                value=default_chat_name(st.session_state.messages),
                max_chars=100,
                key="chat_name_input"
            )

            col_save, col_cancel = st.columns(2)

            with col_save:

                if st.button(
                    "💾 Save & End Chat",
                    type="primary",
                    use_container_width=True
                ):

                    final_name = (
                        chat_name.strip()
                        or default_chat_name(st.session_state.messages)
                    )

                    last = st.session_state.last_decision or {}

                    try:

                        save_chat_to_db(
                            final_name,
                            st.session_state.messages,
                            st.session_state.chat_status,
                            last.get("intent"),
                            last.get("urgency")
                        )

                        st.session_state.saved_chat_name = final_name
                        st.session_state.ending_chat = False
                        st.session_state.end_chat_pending = True
                        st.rerun()

                    except sqlite3.Error as e:

                        st.error(f"Could not save the chat: {e}")

            with col_cancel:

                if st.button(
                    "Cancel",
                    use_container_width=True
                ):

                    st.session_state.ending_chat = False
                    st.rerun()

# ============================================================
# CHAT HISTORY
# ============================================================
elif page == "Chat History":
    st.title("📚 Chat History")
    st.write(
        "View chats that were saved when they were ended."
    )
    st.divider()
    search = st.text_input(
        "🔍 Search by chat name",
        placeholder="Type to filter..."
    )
    chats = list_chats(search.strip())

    if not chats:

        if search.strip():
            st.info("No chats match your search.")
        else:
            st.info("No saved chats yet. End a chat to save it here.")

    else:

        st.write(f"Saved chats: **{len(chats)}**")

        labels = {
            chat["id"]: (
                f"{chat['name']} — {chat['created_at']} "
                f"({chat['message_count']} messages)"
            )
            for chat in chats
        }

        selected_id = st.selectbox(
            "Select a chat",
            options=list(labels.keys()),
            format_func=lambda chat_id: labels[chat_id]
        )

        selected = next(
            chat for chat in chats if chat["id"] == selected_id
        )

        st.divider()

        st.subheader(f"💬 {selected['name']}")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Saved on", selected["created_at"])

        with col2:
            st.metric("Intent", selected["intent"] or "—")

        with col3:
            st.metric("Urgency", selected["urgency"] or "—")

        with col4:
            st.metric(
                "Outcome",
                STATUS_LABELS.get(
                    selected["status"],
                    selected["status"] or "—"
                )
            )

        with st.container(border=True):

            # for st_message in get_chat_messages(selected_id):
            #
            #     with st.chat_message(st_message["role"]):
            #         st.write(st_message["content"])

            for i,chat_message in enumerate(get_chat_messages(selected_id)):
                message(
                    chat_message["content"],
                    is_user=(chat_message["role"] == "user"),
                    key = f"chat_message_{i}"
                )

        st.divider()

        st.subheader("✏️ Manage chat")

        new_name = st.text_input(
            "Rename chat",
            value=selected["name"],
            max_chars=100,
            key=f"rename_{selected_id}"
        )

        col_rename, col_delete = st.columns(2)

        with col_rename:

            if st.button(
                "💾 Save name",
                key=f"save_name_{selected_id}",
                use_container_width=True
            ):

                if new_name.strip():
                    rename_chat(selected_id, new_name.strip())
                    st.rerun()
                else:
                    st.error("Chat name cannot be empty.")

        with col_delete:

            if st.button(
                "🗑️ Delete chat",
                key=f"delete_chat_{selected_id}",
                use_container_width=True
            ):

                delete_chat(selected_id)
                st.rerun()

#=============================================================
# API MANAGEMENT
# ============================================================
elif page == "APIs":
    st.title("🔑 Setup JEV AI API Keys")
    st.write(
        "You can enter multiple JEV API keys "
        "and select one as the default key."
    )
    st.divider()
    # ========================================================
    # ADD API KEY
    # ========================================================

    st.subheader("➕ Add JEV API Key")

    with st.form("add_api_key_form"):

        key_name = st.text_input(
            "API Key Name",
            placeholder="Example: Production Key"
        )

        provider = st.text_input(
            "Provider",
            value="JEV"
        )

        api_key = st.text_input(
            "JEV API Key",
            type="password",
            placeholder="Enter your JEV API key"
        )

        submitted = st.form_submit_button(
            "💾 Save API Key",
            use_container_width=True
        )

        if submitted:

            if not key_name.strip():
                st.error("Please enter a name.")

            elif not api_key.strip():
                st.error("Please enter your JEV API key.")

            else:
                add_api_key(
                    name=key_name.strip(),
                    provider=provider.strip() or "JEV",
                    api_key=api_key.strip()
                )

                st.success(f"✅ API key **{key_name}** saved.")

                st.rerun()

    st.divider()

    # ========================================================
    # SAVED API KEYS
    # ========================================================

    st.subheader("🔐 Saved API Keys")

    api_keys = load_api_keys()

    if not api_keys:

        st.info("No API keys have been added yet.")

    else:

        st.write(f"Total API keys: **{len(api_keys)}**")

        for key in api_keys:

            key_id = key.get("id")
            saved_name = key.get("name", "Unnamed Key")
            saved_provider = key.get("provider", "JEV")
            actual_api_key = key.get("api_key", "")
            is_default = key.get("is_default", False)

            with st.container(border=True):

                col1, col2, col3 = st.columns([3, 3, 2])

                with col1:

                    st.markdown(f"### 🔑 {saved_name}")

                    st.write(f"**Provider:** {saved_provider}")

                    st.code(
                        mask_api_key(actual_api_key),
                        language=None
                    )

                with col2:

                    if is_default:

                        st.success("⭐ DEFAULT API KEY")

                    else:

                        st.write("Not default")

                        if st.button(
                            "⭐ Set as Default",
                            key=f"default_{key_id}",
                            use_container_width=True
                        ):
                            set_default_api_key(key_id)
                            st.rerun()

                with col3:

                    st.write("")

                    if st.button(
                        "🗑️ Delete",
                        key=f"delete_{key_id}",
                        use_container_width=True
                    ):
                        delete_api_key(key_id)
                        st.rerun()

    st.divider()

    # ========================================================
    # CURRENT DEFAULT
    # ========================================================

    st.subheader("⭐ Current Default API")

    default_key = get_default_api_key()

    if default_key:

        st.success(
            f"**{default_key.get('name')}** "
            "is currently selected."
        )

        st.write(
            f"**Provider:** "
            f"{default_key.get('provider', 'JEV')}"
        )

        st.write(
            f"**API Key:** "
            f"{mask_api_key(default_key.get('api_key', ''))}"
        )

    else:

        st.warning("No default API key selected.")

    st.divider()

    st.subheader("🔒 Security")

    st.warning(
        """
        **Important:** API keys are currently stored in
        `api_keys.json`.

        This is suitable for local development/testing.
        For production, use environment variables,
        Streamlit Secrets, an encrypted database,
        or a secrets manager.
        """
    )