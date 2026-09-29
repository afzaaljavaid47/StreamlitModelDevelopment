import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import uuid
import requests
from streamlit_local_storage import LocalStorage
# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Learners Chat Bot Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# FILE CONFIGURATION
# ============================================================

API_KEYS_FILE = "api_keys.json"

DEFAULT_GREETING = [
    {
        "role": "assistant",
        "content": (
            "Hello! 👋 I am your AI Learners "
            "Chat Bot Assistant. How can I help you?"
        )
    }
]


# ============================================================
# API KEY FUNCTIONS
# ============================================================

def load_api_keys():
    """
    Load saved API keys from api_keys.json.
    """

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
    """
    Save API keys to api_keys.json.
    """

    with open(API_KEYS_FILE, "w", encoding="utf-8") as file:
        json.dump(api_keys, file, indent=4)

def mask_api_key(api_key):
    """
    Hide most of the API key when displaying it.
    """

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
    """
    Return the current default API key.
    """

    api_keys = load_api_keys()

    for key in api_keys:
        if key.get("is_default", False):
            return key

    return None

def add_api_key(name, provider, api_key):
    """
    Add a new API key.
    """

    api_keys = load_api_keys()

    # If this is the first key, automatically make it default
    is_first_key = len(api_keys) == 0

    new_key = {
        "id": str(uuid.uuid4()),
        "name": name,
        "provider": provider,
        "api_key": api_key,
        "is_default": is_first_key
    }

    api_keys.append(new_key)

    save_api_keys(api_keys)

def set_default_api_key(key_id):
    """
    Set one API key as the default key.
    """

    api_keys = load_api_keys()

    for key in api_keys:
        key["is_default"] = (
            key.get("id") == key_id
        )

    save_api_keys(api_keys)

def delete_api_key(key_id):
    """
    Delete an API key.
    """

    api_keys = load_api_keys()

    deleted_key = None
    remaining_keys = []

    for key in api_keys:

        if key.get("id") == key_id:
            deleted_key = key
        else:
            remaining_keys.append(key)

    # If deleted key was default,
    # make another key default
    if deleted_key and deleted_key.get("is_default"):

        if remaining_keys:
            remaining_keys[0]["is_default"] = True

    save_api_keys(remaining_keys)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Settings")

page = st.sidebar.radio(
    "Choose Section",
    [
        "Live Chat Support",
        "APIs"
    ]
)

# ============================================================
# LIVE CHAT SUPPORT
# ============================================================
def CallJevAPI(user_message):
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

    payload = {
        "state": {
            "customer_message": user_message
        },
        "model": "jev-latest",
        "questions" :{

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

            "instructions": (
                "How urgent is this customer issue?"
            ),

            "criteria": [
                "Low",
                "Medium",
                "High",
                "Critical"
            ]
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
    }
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
            "error": f"JEV API HTTP error: {response.status_code}",
            "response": response.text
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

if page == "Live Chat Support":

    st.title("🤖 AI Learners Chat Bot Assistant")

    st.write(
        "Welcome to the AI Learners live chat support."
    )

    st.divider()

    # --------------------------------------------------------
    # Local Storage
    # --------------------------------------------------------

    local_storage = LocalStorage()

    # --------------------------------------------------------
    # Check API configuration
    # --------------------------------------------------------

    default_key = get_default_api_key()

    if default_key:

        st.success(
            f"✅ Default API configured: "
            f"**{default_key.get('name', 'Unnamed Key')}**"
        )

        st.caption(
            f"Provider: {default_key.get('provider', 'JEV')} | "
            f"Key: {mask_api_key(default_key.get('api_key', ''))}"
        )

    else:

        st.warning(
            "⚠️ No JEV API key has been configured yet."
        )

        st.info(
            "Go to **APIs** from the sidebar and add your JEV API key."
        )

    st.divider()

    # --------------------------------------------------------
    # Chat UI
    # --------------------------------------------------------

    st.subheader("💬 Live Chat")

    # --------------------------------------------------------
    # Load messages from browser localStorage
    # --------------------------------------------------------

    # LocalStorage() already fetched every stored item into memory when it
    # was constructed above (it blocks briefly on first run to do this), so
    # getItem() here just reads from that in-memory dict. It only takes the
    # item's key - it does NOT accept a `key=` keyword argument.
    stored_messages_raw = local_storage.getItem("chat_messages")

    if "messages" not in st.session_state:

        loaded_messages = None

        if stored_messages_raw:
            try:
                parsed = json.loads(stored_messages_raw)
                if isinstance(parsed, list) and len(parsed) > 0:
                    loaded_messages = parsed
            except (json.JSONDecodeError, TypeError):
                loaded_messages = None

        st.session_state.messages = (
            loaded_messages if loaded_messages else list(DEFAULT_GREETING)
        )

    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User sends message
    user_message = st.chat_input("Type your message...")

    if user_message:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_message
        })

        with st.chat_message("user"):
            st.write(user_message)

        urgency_levels = {
            1: "Low",
            2: "Medium",
            3: "High",
            4: "Critical"
        }
        # AI response
        assistant_response = CallJevAPI(user_message)
        # if assistant_response['success']:
        #     assistant_response=assistant_response["data"]["answers"]
        #     intent = assistant_response["intent"]["choice"]
        #     urgency_score = round(assistant_response["urgency"]["score"])
        #     needs_human = "Yes" if assistant_response["needs_human"]["noul"] * 100 > 75 else "No"
        #     urgency = urgency_levels.get(
        #         urgency_score,
        #         "Unknown"
        #     )
        #
        #     assistant_response = {
        #         "intent": intent,
        #         "urgency": urgency,
        #         "needs_human": needs_human
        #     }
        # else:
        #     assistant_response=assistant_response['response']

        # Add AI response
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_response
        })

        with st.chat_message("assistant"):
            st.write(assistant_response)

        # Save ONLY after a new message
        local_storage.setItem(
            "chat_messages",
            json.dumps(st.session_state.messages),
            key=f"save_chat_messages_{len(st.session_state.messages)}"
        )

    # --------------------------------------------------------
    # End Chat
    # --------------------------------------------------------

    st.divider()

    if st.button(
        "🔴 End Chat",
        use_container_width=True
    ):

        # Get complete conversation
        messages = st.session_state.messages

        # TODO:
        # Save `messages` to your database here.
        #
        # Example:
        #
        # save_conversation(messages)

        # Clear browser localStorage
        local_storage.eraseItem("chat_messages", key="delete_chat_messages")

        # Clear Streamlit session
        st.session_state.messages = list(DEFAULT_GREETING)

        st.success(
            "✅ Chat ended successfully."
        )

# ============================================================
# API MANAGEMENT
# ============================================================

elif page == "APIs":
    st.title("🔑 Setup JEV AI API Keys")
    st.write(
        "You can enter multiple JEV API keys and select "
        "one as the default key."
    )

    st.divider()

    # ========================================================
    # ADD NEW API KEY
    # ========================================================

    st.subheader("➕ Add JEV API Key")

    with st.form("add_api_key_form"):

        key_name = st.text_input(
            "API Key Name",
            placeholder="Example: Production Key"
        )

        provider = st.text_input(
            "Provider",
            value="JEV",
            placeholder="JEV"
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
                st.error(
                    "Please enter a name for the API key."
                )
            elif not api_key.strip():
                st.error(
                    "Please enter your JEV API key."
                )
            else:
                add_api_key(
                    name=key_name.strip(),
                    provider=provider.strip() or "JEV",
                    api_key=api_key.strip()
                )
                st.success(
                    f"✅ API key **{key_name}** saved successfully."
                )
                st.rerun()
    st.divider()

    # ========================================================
    # SAVED API KEYS
    # ========================================================

    st.subheader("🔐 Saved API Keys")

    api_keys = load_api_keys()

    if not api_keys:

        st.info(
            "No API keys have been added yet."
        )

    else:

        st.write(
            f"Total API keys: **{len(api_keys)}**"
        )

        for key in api_keys:

            key_id = key.get("id")
            key_name = key.get(
                "name",
                "Unnamed Key"
            )

            provider = key.get(
                "provider",
                "JEV"
            )

            actual_api_key = key.get(
                "api_key",
                ""
            )

            is_default = key.get(
                "is_default",
                False
            )

            # ------------------------------------------------
            # Card
            # ------------------------------------------------

            with st.container(border=True):

                col1, col2, col3 = st.columns(
                    [3, 3, 2]
                )

                # --------------------------------------------
                # Information
                # --------------------------------------------

                with col1:

                    st.markdown(
                        f"### 🔑 {key_name}"
                    )

                    st.write(
                        f"**Provider:** {provider}"
                    )

                    st.code(
                        mask_api_key(actual_api_key),
                        language=None
                    )

                # --------------------------------------------
                # Default status
                # --------------------------------------------

                with col2:

                    if is_default:

                        st.success(
                            "⭐ DEFAULT API KEY"
                        )

                    else:

                        st.write(
                            "Not default"
                        )

                    if is_default:

                        st.caption(
                            "This API key will be used "
                            "for live chat requests."
                        )

                    else:

                        if st.button(
                            "⭐ Set as Default",
                            key=f"default_{key_id}",
                            use_container_width=True
                        ):

                            set_default_api_key(
                                key_id
                            )

                            st.success(
                                f"**{key_name}** is now "
                                "the default API key."
                            )

                            st.rerun()

                # --------------------------------------------
                # Delete
                # --------------------------------------------

                with col3:

                    st.write("")

                    if st.button(
                        "🗑️ Delete",
                        key=f"delete_{key_id}",
                        type="secondary",
                        use_container_width=True
                    ):

                        delete_api_key(
                            key_id
                        )

                        st.success(
                            f"API key **{key_name}** deleted."
                        )

                        st.rerun()

    st.divider()

    # ========================================================
    # CURRENT DEFAULT API KEY
    # ========================================================

    st.subheader("⭐ Current Default API")

    default_key = get_default_api_key()

    if default_key:

        st.success(
            f"**{default_key.get('name', 'Unnamed Key')}** "
            "is currently selected."
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"**Provider:** "
                f"{default_key.get('provider', 'JEV')}"
            )

        with col2:

            st.write(
                f"**API Key:** "
                f"{mask_api_key(default_key.get('api_key', ''))}"
            )

    else:

        st.warning(
            "No default API key selected."
        )

    st.divider()

    # ========================================================
    # SECURITY INFORMATION
    # ========================================================

    st.subheader("🔒 Security")

    st.warning(
        """
        **Important:** API keys are currently stored in
        `api_keys.json`.

        This is suitable for local development/testing,
        but for production you should use a secure storage
        solution such as environment variables, Streamlit
        Secrets, an encrypted database, or a secrets manager.
        """
    )