from dotenv import load_dotenv
import os
import logging
import random
import json
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database configuration
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# Log database configuration (mask password)
logger.debug(f"DB_USERNAME: {DB_USERNAME}")
logger.debug(f"DB_HOST: {DB_HOST}")
logger.debug(f"DB_PORT: {DB_PORT}")
logger.debug(f"DB_NAME: {DB_NAME}")
logger.debug(f"DB_PASSWORD: {'***' if DB_PASSWORD else None}")

# Validate database configuration
missing_vars = [var for var, value in [
    ("DB_USERNAME", DB_USERNAME),
    ("DB_PASSWORD", DB_PASSWORD),
    ("DB_HOST", DB_HOST),
    ("DB_PORT", DB_PORT),
    ("DB_NAME", DB_NAME)
] if not value]
if missing_vars:
    logger.error(f"Missing database configuration: {', '.join(missing_vars)}")
    raise ValueError(f"Database configuration is incomplete: {missing_vars}")

# Database URL
DATABASE_URL = f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK resources: {e}")
    raise

# Initialize FastAPI app
app = FastAPI()

# CORS configuration
origins = [
    "http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:3003",
    "http://localhost:6001", "http://localhost:6002", "http://localhost:6003",
    "http://localhost:6004", "http://localhost:6005", "https://vliv.app", "https://hrms.vliv.app",
    "https://klms.vliv.app", "https://dms.vliv.app", "https://pms.vliv.app", "https://www.vliv.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize in-memory structures
QueryDesk_AI_Learning = defaultdict(str)
QueryDesk_AI_Response = defaultdict(list)
training_data = []
vectorizer = TfidfVectorizer()
stop_words = set(stopwords.words('english'))
last_retrain_time = datetime.min
table_schema = "public"
used_responses = defaultdict(lambda: defaultdict(set))  # Track used responses

# Intent templates
intent_templates = {
    "cancellation": "Cancel via Manage My Booking: https://shop.flixbus.in/rebooking/login",
    "booking": "Book your ticket here: https://shop.flixbus.in/",
    "general": "For more information, visit our help center: https://help.flixbus.in/"
}

# Query response map (reduced priority)
query_response_map = {
    "cancel ticket": intent_templates["cancellation"],
    "book ticket": intent_templates["booking"],
    "help": intent_templates["general"]
}

# Response map
def load_response_map():
    """Load predefined responses for query categories."""
    return {
        "greeting": {
            "keywords": ["hello", "hi", "greetings", "welcome"],
            "responses": [
                "Hello and welcome to Flix! How may I assist you today?",
                "Greetings! How can I help you with your Flix journey?",
                "Hi, welcome to Flix! I'm here to assist you."
            ]
        },
        "appreciation": {
            "keywords": ["thank you", "thanks", "reaching out"],
            "responses": [
                "Thank you for reaching out! I'm happy to assist.",
                "I appreciate your query. How can I help further?",
                "Thanks for contacting us! Let me assist you."
            ]
        },
        "verification_details": {
            "keywords": ["booking number", "pnr", "passenger name", "verify"],
            "responses": [
                "Please provide your booking number, passenger name, and email/phone to assist you promptly.",
                "To help you, I need your booking number, full name, and contact details."
            ]
        },
        "processing_time": {
            "keywords": ["checking", "moment", "please wait"],
            "responses": [
                "Thanks for the details. Give me a moment to check.",
                "Please wait a moment while I verify your information."
            ]
        },
        "extended_processing": {
            "keywords": ["still checking", "more time"],
            "responses": [
                "I'm still checking your details. Please stay with me.",
                "I need a bit more time to verify. Thanks for your patience."
            ]
        },
        "patience_appreciation": {
            "keywords": ["patience", "thank you for waiting"],
            "responses": [
                "Thank you for your patience!",
                "I appreciate you waiting. Let’s resolve this."
            ]
        },
        "board_from_other_location": {
            "keywords": ["board from other", "change boarding", "different abstr"],
            "responses": [
                "Boarding from a different location is not possible as per our T&C policy. Please use the designated point. Link: https://help.flixbus.com/s/article/PSSPCan-I-board-the-bus-at-a-later-stop",
                "The QR code must be scanned at the specified boarding point. Please board there. Link: https://help.flixbus.com/s/article/PSSPCan-I-board-the-bus-at-a-later-stop"
            ]
        },
        "boarding_point_details": {
            "keywords": ["boarding point", "where to board"],
            "responses": [
                "Your ticket includes boarding point details, route number, and a GPS link for the location.",
                "Check your ticket for boarding point, route number, and GPS link."
            ]
        },
        "price_difference": {
            "keywords": ["price difference", "why price change", "fare change"],
            "responses": [
                "Prices are dynamic and may change based on demand. Book early for the best rates.",
                "Our price lock feature reserves fares for 10 minutes during booking."
            ]
        },
        "pre_departure_call": {
            "keywords": ["pre departure call", "call before departure"],
            "responses": [
                "Pre-departure calls are not mandatory, but the host may contact you before arrival."
            ]
        },
        "bus_delay": {
            "keywords": ["bus late", "bus delay", "delayed bus"],
            "responses": [
                "I’m sorry for the delay due to operational reasons. The bus will reach your point soon.",
                "Apologies for the delay caused by traffic. We’re working to reach you quickly."
            ]
        },
        "missed_bus": {
            "keywords": ["missed bus", "miss bus", "left behind"],
            "responses": [
                "I’m sorry you missed the bus. We must maintain schedules. Please provide booking details for assistance.",
                "We value you but must depart on time. Share your booking number for help."
            ]
        },
        "bus_host_number": {
            "keywords": ["bus number", "host number", "driver number"],
            "responses": [
                "I can’t provide the bus number, but your ticket’s route number is displayed on the bus.",
                "We don’t share driver contacts. Please be at the pickup point 15 minutes early."
            ]
        },
        "where_is_bus": {
            "keywords": ["where is bus", "bus location", "track bus"],
            "responses": [
                "The bus will arrive as per your ticket’s schedule. Check the route number and tracking link: https://global.flixbus.com/track/order/3242800682",
                "Please be at the boarding point 15 minutes early. Track here: https://global.flixbus.com/track/order/3242800682"
            ]
        },
        "where_is_boarding_point": {
            "keywords": ["boarding point location", "where is boarding point"],
            "responses": [
                "Your boarding point address and Google Maps link are on your ticket. Arrive 15 minutes early."
            ]
        },
        "pax_running_late": {
            "keywords": ["running late", "late to board", "wait for me"],
            "responses": [
                "The bus departs on schedule. Please reach the boarding point before the ticketed time.",
                "Buses can’t wait due to schedules. Cancel up to 15 minutes before via: https://shop.flixbus.in/rebooking/login"
            ]
        },
        "ride_cancellation": {
            "keywords": ["ride cancelled", "bus cancelled", "cancellation"],
            "responses": [
                "Sorry for the cancellation. A self-help link was emailed to rebook or refund: https://shop.flixbus.in/rebooking/login",
                "I can process a full refund for the cancelled ride. Shall I proceed?"
            ]
        },
        "no_show_refund_denial": {
            "keywords": ["no show", "missed bus refund", "didn't board"],
            "responses": [
                "Refunds aren’t possible for no-shows per our policy: https://www.flixbus.in/terms-and-conditions, clause 12.2.5",
                "The bus departed after others boarded, so we can’t refund: https://www.flixbus.in/terms-and-conditions, clause 12.2.5"
            ]
        },
        "booking_changes_denial": {
            "keywords": ["change booking", "modify booking"],
            "responses": [
                "Bookings can’t be modified here. Use Manage My Booking: https://shop.flixbus.in/rebooking/login",
                "Please update your booking via: https://shop.flixbus.in/rebooking/login"
            ]
        },
        "booking_process": {
            "keywords": ["how to book", "booking process", "book ticket"],
            "responses": [
                "Book via our website: select seats, enter passenger/contact details, pay via card/UPI. Includes 7kg hand luggage, 20kg regular free."
            ]
        },
        "manage_booking_changes": {
            "keywords": ["change date", "change time", "cancel ticket"],
            "responses": [
                "Change or cancel via Manage My Booking: https://shop.flixbus.in/rebooking/login",
                "Reschedule or cancel your ride here: https://shop.flixbus.in/rebooking/login"
            ]
        },
        "complaint_feedback": {
            "keywords": ["complain", "feedback", "review"],
            "responses": [
                "Thank you for your feedback! We’re glad to hear from you."
            ]
        },
        "rude_behavior": {
            "keywords": ["rude driver", "rude host", "bad behavior"],
            "responses": [
                "I’m sorry for the rude behavior. I’ll escalate this for review and action."
            ]
        },
        "breakdown_refunded": {
            "keywords": ["bus breakdown", "ac not working", "refund breakdown"],
            "responses": [
                "Sorry for the breakdown. Please share booking details for assistance."
            ]
        },
        "route_details": {
            "keywords": ["route details", "bus route"],
            "responses": [
                "We have stop locations for your booking. Need anything else?"
            ]
        },
        "change_date": {
            "keywords": ["change date", "reschedule date"],
            "responses": [
                "Change your date up to 15 minutes before departure: https://shop.flixbus.in/rebooking/login"
            ]
        },
        "route_information": {
            "keywords": ["route information", "bus route info"],
            "responses": [
                "View route details for booked tickets: https://www.flixbus.in/track/"
            ]
        },
        "flix_lounge": {
            "keywords": ["flix lounge", "anand vihar lounge"],
            "responses": [
                "No Flix Lounge at Anand Vihar. Please wait at the boarding point."
            ]
        },
        "bus_delay_less_120": {
            "keywords": ["bus delay less than 120", "short delay"],
            "responses": [
                "Sorry for the delay under 120 minutes. Refunds apply only for delays over 120 minutes: https://www.flixbus.in/terms-and-conditions"
            ]
        },
        "bus_delay_over_120": {
            "keywords": ["bus delay over 120", "long delay"],
            "responses": [
                "For delays over 2 hours, I can refund your ticket. Proceed?"
            ]
        },
        "bus_breakdown_ac": {
            "keywords": ["ac not working", "bus breakdown ac"],
            "responses": [
                "Sorry for the AC issue. Share booking details for resolution."
            ]
        },
        "luggage_policy": {
            "keywords": ["luggage policy", "baggage rules"],
            "responses": [
                "Free: 7kg hand, 20kg regular luggage. Add more: https://shop.flixbus.in/rebooking/login"
            ]
        },
        "cancel_ticket": {
            "keywords": ["cancel ticket", "ticket cancellation"],
            "responses": [
                "Cancel up to 15 minutes before departure: https://shop.flixbus.in/rebooking/login"
            ]
        },
        "stranded_passenger": {
            "keywords": ["stranded", "left behind"],
            "responses": [
                "Sorry for the issue. Share booking details; no refunds per policy."
            ]
        },
        "lost_item": {
            "keywords": ["lost item", "left something", "lost and found"],
            "responses": [
                "Please fill out our Lost and Found form to recover your items."
            ]
        },
        "travel_with_pet": {
            "keywords": ["travel with pet", "pet policy"],
            "responses": [
                "Pets aren’t allowed on Flix buses for safety reasons."
            ]
        },
        "prices_discounts": {
            "keywords": ["price", "discount", "offer"],
            "responses": [
                "Your ticket price is final. No discounts currently available.",
                "Book early for the best dynamic prices."
            ]
        },
        "blanket_service": {
            "keywords": ["blanket", "blanket service"],
            "responses": [
                "No blankets provided; bring your own for comfort."
            ]
        },
        "water_bottle_service": {
            "keywords": ["water", "water bottle"],
            "responses": [
                "No water bottles provided. Please bring your own."
            ]
        },
        "washroom_service": {
            "keywords": ["washroom", "restroom", "toilet"],
            "responses": [
                "No washrooms on buses, but breaks are scheduled."
            ]
        },
        "seat_changes": {
            "keywords": ["change seat", "seat change"],
            "responses": [
                "Seats are auto-assigned and can’t be changed."
            ]
        },
        "shadow_booking": {
            "keywords": ["shadow booking", "payment not found", "booking not found"],
            "responses": [
                "No booking found. Provide passenger name, email, phone, and payment proof."
            ]
        },
        "no_refunded_statement": {
            "keywords": ["no refund", "refund denial"],
            "responses": [
                "No refund possible as the bus departed with others boarded."
            ]
        },
        "refund_processing": {
            "keywords": ["refund status", "refund processing"],
            "responses": [
                "Refund of (AMOUNT) from (DATE) will be credited in 7 days."
            ]
        },
        "refund_tat_crossed": {
            "keywords": ["refund not received", "late refund"],
            "responses": [
                "Refund initiated on [DATE]. Check with your bank or share statement."
            ]
        },
        "closing_statement": {
            "keywords": ["goodbye", "bye", "thanks", "done"],
            "responses": [
                "Thanks for contacting Flix! Have a great day!",
                "Happy to help! Reach out anytime."
            ]
        },
        "request_feedback": {
            "keywords": ["feedback", "rate conversation", "survey"],
            "responses": [
                "Please share feedback via the link after our chat."
            ]
        }
    }

response_map = load_response_map()

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    companyemail: str
    companyid: int

class QueryResponse(BaseModel):
    query: str
    companyemail: str
    companyid: int
    response: str

class HealthResponse(BaseModel):
    status: str

# Database connection
try:
    engine = create_engine(DATABASE_URL, connect_args={"options": "-c default_transaction_read_only=on"})
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
        logger.info("Database connection established successfully")
except SQLAlchemyError as e:
    logger.error(f"Database connection failed: {e}")
    raise

# Helper functions
def find_table_schema(table_name: str) -> Optional[str]:
    """Find the schema containing the specified table."""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("""
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            """))
            tables = [(row[0], row[1]) for row in result]
            logger.debug(f"Available tables: {tables}")
            for schema, tbl in tables:
                if tbl.lower() == table_name.lower():
                    logger.info(f"Found table {table_name} in schema {schema}")
                    return schema
            logger.warning(f"Table {table_name} not found in any schema")
            return None
    except SQLAlchemyError as e:
        logger.error(f"Error finding schema for {table_name}: {e}")
        return None

def check_table_exists():
    """Check if QueryStatements table exists."""
    global table_schema
    table_schema = "public"
    try:
        with engine.connect() as connection:
            result = connection.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'QueryStatements'
            """))
            columns = [row[0] for row in result]
            if columns:
                logger.info(f"Table QueryStatements found in schema public with columns: {columns}")
                return True
            logger.error("Table QueryStatements not found in schema public")
            return False
    except SQLAlchemyError as e:
        logger.error(f"Error checking QueryStatements table: {e}")
        return False

def preprocess_text(text: str) -> str:
    """Preprocess text for TF-IDF."""
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    return ' '.join(tokens)

def generate_fallback_response(query: str) -> str:
    """Generate a response based on query tokens."""
    tokens = word_tokenize(query.lower())
    if not tokens:
        return f"I'm not sure how to assist with an empty query. Please visit our help center: {intent_templates['general']}"
    if any(word in tokens for word in ["cancel", "cancellation"]):
        return f"To cancel your booking, please visit: {intent_templates['cancellation']}"
    elif any(word in tokens for word in ["book", "booking", "ticket"]):
        return f"To make a booking, please visit: {intent_templates['booking']}"
    elif any(word in tokens for word in ["bus", "travel", "trip"]):
        return "Thank you for sharing the details. Please allow me a moment to look for your concern."
    else:
        return f"I'm not sure how to assist with '{query}'. Please visit our help center: {intent_templates['general']}"

def generate_ai_response(query: str, category: str, used_responses_set: set) -> str:
    """Generate a new AI response for the category, avoiding used responses."""
    tokens = word_tokenize(query.lower())
    base_responses = {
        "greeting": "Welcome to Flix! How can I assist you with your travel needs?",
        "appreciation": "I’m grateful for your query. How can I assist you further?",
        "verification_details": "Could you share your booking number and contact details for quick assistance?",
        "processing_time": "Just a moment while I check your request details.",
        "extended_processing": "I’m still verifying your details. Thanks for waiting!",
        "patience_appreciation": "Thanks for being patient while I assist you!",
        "board_from_other_location": "Please board at the designated point as per our policy.",
        "boarding_point_details": "Your boarding point and GPS link are available on your ticket.",
        "price_difference": "Ticket prices vary based on demand. Early booking secures better rates.",
        "pre_departure_call": "The host may call before departure, but it’s not guaranteed.",
        "bus_delay": "Sorry for the bus delay. It should arrive shortly.",
        "missed_bus": "Sorry you missed the bus. Please share booking details for next steps.",
        "bus_host_number": "Your ticket’s route number is shown on the bus. Arrive early!",
        "where_is_bus": "Track your bus using the link on your ticket.",
        "where_is_boarding_point": "Check your ticket for the boarding point’s address and map link.",
        "pax_running_late": "Please arrive on time, as buses depart per schedule.",
        "ride_cancellation": "Sorry for the cancelled ride. Check your email for rebooking options.",
        "no_show_refund_denial": "No refunds for no-shows, per our terms.",
        "booking_changes_denial": "Modify your booking online via our rebooking portal.",
        "booking_process": "Book tickets online with your preferred seat and payment method.",
        "manage_booking_changes": "Update or cancel your booking through our online portal.",
        "complaint_feedback": "Thanks for your feedback! It helps us improve.",
        "rude_behavior": "Apologies for any rudeness. I’ll report this for investigation.",
        "breakdown_refunded": "Sorry for the bus issue. Please provide booking details.",
        "route_details": "Your booking includes all stop locations.",
        "change_date": "Reschedule your trip online up to 15 minutes before departure.",
        "route_information": "Check route details online for your booked trip.",
        "flix_lounge": "No lounge at this location. Please wait at the boarding point.",
        "bus_delay_less_120": "Apologies for the short delay. Refunds apply for delays over 120 minutes.",
        "bus_delay_over_120": "For long delays, you may be eligible for a refund. Shall I assist?",
        "bus_breakdown_ac": "Sorry for the AC problem. Please share your booking details.",
        "luggage_policy": "You get 7kg hand and 20kg regular luggage free.",
        "cancel_ticket": "Cancel your ticket online before departure.",
        "stranded_passenger": "Sorry for the inconvenience. Please provide booking details.",
        "lost_item": "Submit a Lost and Found form to locate your item.",
        "travel_with_pet": "Pets are not permitted on our buses for safety.",
        "prices_discounts": "Prices are fixed at booking. Check for future offers.",
        "blanket_service": "Bring a blanket for your comfort.",
        "water_bottle_service": "Bring your own water for the journey.",
        "washroom_service": "Buses lack washrooms, but breaks are planned.",
        "seat_changes": "Seats are fixed at booking and cannot be changed.",
        "shadow_booking": "No booking found. Please share payment proof and details.",
        "no_refunded_statement": "Refund not possible due to departure.",
        "refund_processing": "Your refund is being processed and will reflect soon.",
        "refund_tat_crossed": "Check your bank for the refund or share your statement.",
        "closing_statement": "Thanks for reaching out! Safe travels!",
        "request_feedback": "We’d love your feedback to improve our service."
    }
    new_response = base_responses.get(category, f"Thank you for your query about '{query}'. Please provide more details.")
    counter = 1
    while new_response in used_responses_set:
        new_response = f"{base_responses.get(category, 'Thank you for your query.')} (Response {counter})"
        counter += 1
    return new_response

def load_training_data():
    """Load training data from JSON file."""
    global training_data
    try:
        with open("training_data.json", "r") as f:
            data = json.load(f)
            training_data = [preprocess_text(str(stmt)) for stmt in data.get("statements", []) if stmt]
            logger.info(f"Loaded {len(training_data)} statements from training_data.json")
    except FileNotFoundError:
        logger.warning("training_data.json not found, initializing with default statements")
        training_data = [preprocess_text(stmt) for stmt in ["cancel ticket", "book ticket", "bus travel"]]
        with open("training_data.json", "w") as f:
            json.dump({"statements": ["cancel ticket", "book ticket", "bus travel"]}, f, indent=2)
    except json.JSONDecodeError:
        logger.error("Invalid JSON in training_data.json, resetting to default")
        training_data = [preprocess_text(stmt) for stmt in ["cancel ticket", "book ticket", "bus travel"]]
        with open("training_data.json", "w") as f:
            json.dump({"statements": ["cancel ticket", "book ticket", "bus travel"]}, f, indent=2)
    if training_data:
        retrain_vectorizer()

def save_training_data():
    """Save training data to JSON file."""
    try:
        with open("training_data.json", "w") as f:
            json.dump({"statements": training_data}, f, indent=2)
            logger.info("Saved training data to training_data.json")
    except Exception as e:
        logger.error(f"Error saving training_data.json: {e}")

def check_data_refresh():
    """Refresh training data from database (read-only)."""
    global training_data, table_schema
    if not check_table_exists():
        logger.warning("Skipping database refresh due to missing QueryStatements table")
        return
    try:
        with engine.connect() as connection:
            result = connection.execute(text('SELECT statements FROM public."QueryStatements" WHERE companyid = 331'))
            statements = []
            for row in result:
                json_statements = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                if isinstance(json_statements, list):
                    statements.extend(json_statements)
                else:
                    statements.append(json_statements)
            new_training_data = [preprocess_text(str(stmt)) for stmt in statements if stmt]
            if new_training_data != training_data:
                training_data = new_training_data
                retrain_vectorizer()
                save_training_data()
                logger.info(f"Refreshed training data with {len(training_data)} statements from database")
    except SQLAlchemyError as e:
        logger.warning(f"Skipping database refresh due to error: {e}")

def retrain_vectorizer():
    """Retrain TF-IDF vectorizer."""
    global vectorizer
    if training_data:
        try:
            vectorizer = TfidfVectorizer()
            vectorizer.fit(training_data)
            logger.info("TF-IDF vectorizer retrained")
        except ValueError as e:
            logger.error(f"Error retraining vectorizer: {e}")

def fetch_similar_statements(query: str, top_k: int = 1) -> str:
    """Fetch the most similar statement from database, handling JSON arrays."""
    preprocessed_query = preprocess_text(query)
    if not training_data or not preprocessed_query:
        return ""
    try:
        query_vec = vectorizer.transform([preprocessed_query])
        training_vec = vectorizer.transform(training_data)
        similarities = cosine_similarity(query_vec, training_vec)
        top_index = np.argmax(similarities[0])
        if similarities[0][top_index] > 0.05:
            with engine.connect() as connection:
                result = connection.execute(text('SELECT statements FROM public."QueryStatements" WHERE companyid = 331'))
                statements = []
                for row in result:
                    json_statements = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                    if isinstance(json_statements, list):
                        statements.extend(json_statements)
                    else:
                        statements.append(json_statements)
                return statements[top_index] if top_index < len(statements) else ""
        return ""
    except SQLAlchemyError as e:
        logger.warning(f"Skipping similar statements fetch due to error: {e}")
        return ""
    except Exception as e:
        logger.error(f"Error in fetch_similar_statements: {e}")
        return ""

def learn_phrase(query: str, intent: str):
    """Store query-intent mapping in memory and training data."""
    preprocessed_query = preprocess_text(query)
    if preprocessed_query not in QueryDesk_AI_Learning:
        QueryDesk_AI_Learning[preprocessed_query] = intent
        training_data.append(preprocessed_query)
        retrain_vectorizer()
        save_training_data()
        logger.info(f"Learned new phrase: {query} -> {intent}")

def generate_dynamic_response(query: str, intent: str, companyemail: str, companyid: int) -> str:
    """Generate response, prioritizing database, then response_map, then AI."""
    preprocessed_query = preprocess_text(query)
    session_key = f"{companyemail}_{companyid}"
    similar_statement = fetch_similar_statements(query)
    if similar_statement:
        QueryDesk_AI_Response[query].append({
            "response": similar_statement,
            "companyemail": companyemail,
            "companyid": companyid,
            "timestamp": datetime.utcnow().isoformat()
        })
        learn_phrase(query, intent)
        return similar_statement
    for category, data in response_map.items():
        for keyword in data["keywords"]:
            if keyword in preprocessed_query:
                available_responses = [
                    resp for resp in data["responses"]
                    if resp not in used_responses[session_key][category]
                ]
                if available_responses:
                    response = random.choice(available_responses)
                    used_responses[session_key][category].add(response)
                    QueryDesk_AI_Response[query].append({
                        "response": response,
                        "companyemail": companyemail,
                        "companyid": companyid,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    learn_phrase(query, category)
                    return response
                else:
                    response = generate_ai_response(query, category, used_responses[session_key][category])
                    used_responses[session_key][category].add(response)
                    QueryDesk_AI_Response[query].append({
                        "response": response,
                        "companyemail": companyemail,
                        "companyid": companyid,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    learn_phrase(query, category)
                    return response
    if query.lower() in query_response_map:
        response = query_response_map[query.lower()]
        QueryDesk_AI_Response[query].append({
            "response": response,
            "companyemail": companyemail,
            "companyid": companyid,
            "timestamp": datetime.utcnow().isoformat()
        })
        learn_phrase(query, intent)
        return response
    response = generate_ai_response(query, "general", used_responses[session_key]["general"])
    used_responses[session_key]["general"].add(response)
    QueryDesk_AI_Response[query].append({
        "response": response,
        "companyemail": companyemail,
        "companyid": companyid,
        "timestamp": datetime.utcnow().isoformat()
    })
    learn_phrase(query, intent)
    return response

async def periodic_retrain():
    """Periodically retrain every 24 hours."""
    global last_retrain_time
    while True:
        if datetime.utcnow() - last_retrain_time > timedelta(hours=24):
            check_data_refresh()
            last_retrain_time = datetime.utcnow()
            logger.info("Periodic retraining completed")
        await asyncio.sleep(3600)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_training_data()
    check_data_refresh()
    task = asyncio.create_task(periodic_retrain())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return {"message": "Welcome to the AI-driven Query Response API"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Health check: database connected successfully")
        return {"status": "Connected successfully"}
    except SQLAlchemyError as e:
        logger.error(f"Health check: database connection failed: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

@app.post("/generate-response", response_model=QueryResponse)
async def generate_response(request: QueryRequest):
    if not request.query or not request.companyemail or request.companyid is None:
        raise HTTPException(status_code=400, detail="Request body is empty")
    logger.info(f"Received query: {request.query}, companyemail: {request.companyemail}, companyid: {request.companyid}")
    try:
        preprocessed_query = preprocess_text(request.query)
        intent = QueryDesk_AI_Learning.get(preprocessed_query, "general")
        response = generate_dynamic_response(
            request.query,
            intent,
            request.companyemail,
            request.companyid
        )
        return QueryResponse(
            query=request.query,
            companyemail=request.companyemail,
            companyid=request.companyid,
            response=response
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Error processing query")

@app.get("/view-learned")
async def view_learned():
    try:
        tables = []
        statements = []
        with engine.connect() as connection:
            result = connection.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
            tables = [row[0] for row in result]
            if check_table_exists():
                result = connection.execute(text('SELECT statements FROM public."QueryStatements" WHERE companyid = 331'))
                for row in result:
                    json_statements = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                    if isinstance(json_statements, list):
                        statements.extend(json_statements)
                    else:
                        statements.append(json_statements)
        return {
            "database_tables": tables,
            "database_statements": statements,
            "QueryDesk_AI_response": dict(QueryDesk_AI_Response),
            "QueryDesk_AI_learning": dict(QueryDesk_AI_Learning)
        }
    except SQLAlchemyError as e:
        logger.warning(f"Returning learned data without database statements due to error: {e}")
        return {
            "database_tables": [],
            "database_statements": [],
            "QueryDesk_AI_response": dict(QueryDesk_AI_Response),
            "QueryDesk_AI_learning": dict(QueryDesk_AI_Learning)
        }

@app.get("/view-statements")
async def view_statements():
    try:
        statements = []
        if check_table_exists():
            with engine.connect() as connection:
                result = connection.execute(text('SELECT statements FROM public."QueryStatements" WHERE companyid = 331'))
                for row in result:
                    json_statements = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                    if isinstance(json_statements, list):
                        statements.extend(json_statements)
                    else:
                        statements.append(json_statements)
        return {"QueryStatements": statements}
    except SQLAlchemyError as e:
        logger.warning(f"Returning empty statements due to error: {e}")
        return {"query_statements": []}

if __name__ == "__main__":
    import uvicorn
    check_data_refresh()
    uvicorn.run(app, host="0.0.0.0", port=8000)