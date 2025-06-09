from dotenv import load_dotenv
import os
import logging
import random
import json
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Dict, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager
import difflib
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

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
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK resources: {e}")
    raise

# Initialize FastAPI app
app = FastAPI()

# Custom middleware to handle malformed JSON
class JSONValidationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and request.url.path == "/generate-response":
            try:
                # Attempt to parse JSON body
                body = await request.json()
                # Check for empty or missing fields
                required_fields = ["query", "companyemail", "companyid"]
                if not all(field in body and body[field] for field in required_fields):
                    logger.error(f"Missing or empty fields in request: {body}")
                    return JSONResponse(
                        status_code=200,
                        content={"success": False, "response": "Please fill the required data"}
                    )
            except json.JSONDecodeError as e:
                logger.error(f"Malformed JSON: {str(e)}")
                return JSONResponse(
                    status_code=200,
                    content={"success": False, "response": "Please fill the required data"}
                )
        return await call_next(request)

app.add_middleware(JSONValidationMiddleware)

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

# Initialize in-memory structures
QueryDesk_AI_Learning = defaultdict(str)
QueryDesk_AI_Response = defaultdict(list)
training_data = []
vectorizer = TfidfVectorizer()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
last_retrain_time = datetime.min
table_schema = "public"
used_responses = defaultdict(set)  # Track all used responses per session
db_statements = []  # Cache database statements
training_vec = None  # Cache vectorized training data
training_stats = {"learned_phrases": 0, "last_retrain": None, "training_data_size": 0}

# Intent templates
intent_templates = {
    "cancellation": "Cancel via Manage My Booking: https://shop.flixbus.in/rebooking/login",
    "booking": "Book your ticket here: https://shop.flixbus.in/",
    "general": "For more information, visit our help center: https://help.flixbus.in/",
    "out_of_context": "I'm not sure about that, but I can help with travel queries! Try asking about tickets or buses."
}

# Query response map (lowest priority)
query_response_map = {
    "cancel ticket": intent_templates["cancellation"],
    "book ticket": intent_templates["booking"],
    "help": intent_templates["general"]
}

# Response map
def load_response_map():
    """Load predefined responses for specific query categories."""
    return {
        "greeting": {
            "keywords": ["hello", "hi", "greetings", "welcome"],
            "responses": [
                "Hello and welcome to Flix! I'm here to assist you. How may I help you today?",
                "Greetings and welcome to Flix. How may I assist you?",
                "Welcome to Flix! My name is your assistant. How can I help you?",
                "Hi, Welcome to Flix, I am your assistant. How may I assist you?"
            ]
        },
        "appreciation": {
            "keywords": ["thank you", "reaching out", "contacted", "reporting"],
            "responses": [
                "I appreciate that you have reached out to us with your query.",
                "Thank you for reaching out with your inquiry.",
                "I am grateful that you contacted us regarding your concern.",
                "I appreciate that you are reporting to us about this concern. I will look into it immediately."
            ]
        },
        "verification_details": {
            "keywords": ["booking number", "pnr", "passenger name", "verify"],
            "responses": [
                "Sure, I'm here to help. To assist you effectively, please provide the following details: Booking number or PNR Number, Passenger's full name, Booking email address or booking phone number. Once you share this information, I'll be able to assist you promptly.",
                "Certainly, I'm ready to assist you. To ensure I can help you effectively, please provide the following details: Booking number, Passenger's full name, Booking email address or booking phone number. Once you provide these details, I'll be able to assist you accordingly."
            ]
        },
        "processing_time": {
            "keywords": ["checking", "moment", "please wait"],
            "responses": [
                "Thank you for sharing the details. Please allow me a moment to look for your concern.",
                "Thank you for providing the information. Please allow me a moment to check with the details.",
                "Thank you for sharing these details. Just a moment, please, I'm checking on the same."
            ]
        },
        "extended_processing": {
            "keywords": ["still checking", "more time"],
            "responses": [
                "I am still checking your booking/ride details. Please be with me.",
                "I got booking however need to check the issue with your ride."
            ]
        },
        "patience_appreciation": {
            "keywords": ["patience", "thank you for waiting"],
            "responses": [
                "Thank you so much for your patience.",
                "I really appreciate your patience.",
                "Thank you so much for notifying me about the issue.",
                "Thank you for reaching out to me about this.",
                "I will get your issue resolved positively.",
                "I appreciate your patience in this matter.",
                "Your patience is appreciable."
            ]
        },
        "board_from_other_location": {
            "keywords": ["board from other", "change boarding", "different location"],
            "responses": [
                "May I know why you want to change the boarding point and make it to other? As per the T&C, your ticket is valid for the boarding location to booked for and I regret to inform that boarding from other location isn't possible. Link: https://help.flixbus.com/s/article/PSSPCan-I-board-the-bus-at-a-later-stop?language=en_IN",
                "I understand your preference for boarding the bus from your desired location. However, to ensure a smooth process, we kindly ask you to board from the designated location where the QR code can be scanned by the bus staff. Link: https://help.flixbus.com/s/article/PSSPCan-I-board-the-bus-at-a-later-stop?language=en_IN",
                "While I appreciate your desire to board the bus from a different location, please note that the process requires the QR code to be scanned by the host at the specified boarding point. We recommend using the designated location for a seamless experience. Link: https://help.flixbus.com/s/article/PSSPCan-I-board-the-bus-at-a-later-stop?language=en_IN",
                "I understand your request to board the bus from your chosen location. Unfortunately, due to our procedures, the QR code must be scanned by the host at the specified point. We encourage you to use the designated boarding location for a smooth transition. Link: https://help.flixbus.com/s/article/PSSPCan-I-board-the-bus-at-a-later-stop?language=en_IN",
                "Thank you for your understanding. Although we recognize your preferred boarding location, our process mandates that the QR code be scanned by the host at the assigned area. For a hassle-free experience, please board from the designated location. Link: https://help.flixbus.com/s/article/PSSPCan-I-board-the-bus-at-a-later-stop?language=en_IN"
            ]
        },
        "boarding_point_details": {
            "keywords": ["boarding point", "where to board"],
            "responses": [
                "The ticket contains essential details such as the boarding points, bus route number, a link in the bottom right corner, and a GPS link that can be accessed by clicking on the boarding point for more information.",
                "You can find important information on the ticket, including boarding points, the bus route number, a link at the bottom right corner, and a GPS link accessible by clicking on the boarding point.",
                "The ticket provides various details, such as the boarding points, the bus route number, a link located in the bottom right corner, and a GPS link available by clicking on the boarding point for further information."
            ]
        },
        "price_difference": {
            "keywords": ["price difference", "why price change", "fare change"],
            "responses": [
                "I would like to inform you that as the prices are dynamic in nature and may change on the website with time.",
                "Please note that our prices are dynamically adjusted based on demand, availability, and other factors to ensure the best possible experience for all our passengers.",
                "To provide you with the most accurate and fair pricing, our rates are dynamically updated. We recommend booking early to secure the best available price.",
                "We always recommend our passengers about a price lock feature where the price shown at the time of selection is reserved for 10 minutes, allowing passengers to complete their booking at the same rate within this timeframe. Delays in booking beyond this period may result in price adjustments.",
                "We suggest booking early and utilizing the price lock feature to secure the most favorable fare."
            ]
        },
        "pre_departure_call": {
            "keywords": ["pre departure call", "call before departure"],
            "responses": [
                "I completely understand that you are expecting a call from the bus staff. Please note that pre-departure calls are not mandatory however, the host might call you before arriving at your departure point. We appreciate your understanding."
            ]
        },
        "bus_delay": {
            "keywords": ["bus late", "bus delay", "delayed bus"],
            "responses": [
                "I regret to inform you that the bus is running late due to some operational reason and apologize for the inconvenience caused due to delay. So requesting you to please be available on the boarding point as the ride has already been departed from the previous boarding point and now going to reach your boarding point as quickly as possible. I appreciate your understanding.",
                "I’m sincerely apologizing for the delay of the ride and sorry for the inconvenience caused to you. I would like to inform you that due to some operational reason the ride has been delayed and our operational team is trying to manage the delay and reach out your boarding point as soon as possible.",
                "I regret to inform you that due to the heavy traffic the ride got stuck and now it’s get back on the track and the operational team is trying to manage that delay and reach out to your boarding point as quickly as possible. We need your patience and understanding in this matter."
            ]
        },
        "missed_bus": {
            "keywords": ["missed bus", "miss bus", "left behind"],
            "responses": [
                "We see you as a valued customer and do not wish to leave you behind. However, it’s necessary to move the bus on schedule, as we have a commitment to punctuality. I sincerely apologize for the inconvenience caused by missing the bus.",
                "We appreciate you as a valued customer and have no intention of leaving you behind. At the same time, we must adhere to the bus's scheduled departure time for the sake of punctuality. I understand how frustrating this can be, and I’m here to help you any further assistance you may need.",
                "We truly value you as a customer, and it’s not in our interest to leave you behind. However, we also have to ensure the bus departs on time, as punctuality is essential. Please help me with the booking reference number or the PNR number for verification process so that I can help you with the information associated with your journey with Flix."
            ]
        },
        "bus_host_number": {
            "keywords": ["bus number", "host number", "driver number"],
            "responses": [
                "I really apologize that I am unable to provide the bus number however you may identify your ride through the route number mentioned on the ticket and it gets displayed in front of the bus as well.",
                "I regret to inform you that we don't have access to the bus driver or host's contact information. So, we always request our passengers to be at the pickup point 15 minutes prior to the departure, You may track your ride through the tracing link mentioned on the ticket."
            ]
        },
        "where_is_bus": {
            "keywords": ["where is bus", "bus location", "track bus"],
            "responses": [
                "As checked, the ride is for your booked route. So, I would like to inform you that the ride will arrive at the boarding point as per the mentioned time on the ticket. I’m requesting you to please be present at the boarding point 15 minutes before the departure time of the bus, so that you can easily board the bus. Here I’m sharing the bus route number which is mentioned on the ticket just under the boarding point details and it’s also available in front of the bus from which you can easily recognize your bus. Tracking Link: https://global.flixbus.com/track/order/3242800682",
                "I’m requesting you to please be available on the boarding point 15 minutes before the departure time of the bus, so that you can easily board the bus. Have a safe and pleasant journey ahead with Flix. Tracking Link: https://global.flixbus.com/track/order/3242800682"
            ]
        },
        "where_is_boarding_point": {
            "keywords": ["boarding point location", "where is boarding point"],
            "responses": [
                "As checked the ticket is for your booked route. This is the address of your boarding point. This is the Google map link for the exact boarding point location. I’m requesting you to please be available on the boarding point 15 minutes before the departure time of the bus, so that you can easily board the bus. Have a safe and pleasant journey ahead with Flix."
            ]
        },
        "pax_running_late": {
            "keywords": ["running late", "late to board", "wait for me"],
            "responses": [
                "Extremely sorry to inform you that the bus will depart from the boarding point as per the scheduled time so requesting you to please try to reach the boarding point before the mentioned time on the ticket and ensure not to miss the bus.",
                "Unfortunately, the bus cannot wait for the delayed passengers. Our buses travel within a network and are bound to follow a timetable. Please ensure that you are at the stop at least 15 minutes before departure. If you realize that you’re not going to reach in time, you can cancel your ride up to 15 minutes before departure via manage my booking of our website. Link: https://shop.flixbus.in/rebooking/login"
            ]
        },
        "ride_cancellation": {
            "keywords": ["ride cancelled", "bus cancelled", "cancellation"],
            "responses": [
                "I’m really sorry for the inconvenience caused to you due to the ride cancellation and I would like to inform you that due to some operational reason the ride has been cancelled and after cancellation a self-help link has been provided to you via email with the booking email id. So requesting you to please check the email inbox along with the spam folder for the same and after clicking on that link you will be able to book an alternative ride completely free of cost for any other day you want or you may cancel the trip and generate a full ticket refund for yourself which will be credited to the source account within 7 working days.",
                "I’m really sorry for the inconvenience caused to you due to the ride cancellation. Or if you want I can help you with the full ticket refund by cancelling the ticket from our end and the refund will be credited to the source account within 7 working days excluding Saturday, Sunday and Public holidays/ Bank holidays. With your permission should I proceed with the refund?"
            ]
        },
        "no_show_refund_denial": {
            "keywords": ["no show", "missed bus refund", "didn't board"],
            "responses": [
                "I deeply apologize for the inconvenience you're experiencing, and I understand your frustration. I'm unable to proceed with either the refund or booking an alternative ride for you at this time. As per our company policy, in such cases where the service has been provided as intended, we are unable to process a refund. Link: https://www.flixbus.in/terms-and-conditions-of-carriage, clause number 12.2.5",
                "After thoroughly investigating the incident, we confirmed that the bus arrived at the designated boarding point, other passengers boarded successfully, and the bus departed after the scheduled time. Regrettably, under these circumstances, we are unable to issue a refund for your ticket. Link: https://www.flixbus.in/terms-and-conditions-of-carriage, clause number 12.2.5"
            ]
        },
        "booking_changes_denial": {
            "keywords": ["change booking", "modify booking"],
            "responses": [
                "Once your ticket is booked, we are unable to modify it from our side. Kindly visit our website, go to ‘Manage My Booking,’ and fill in the required information to make changes. Link: https://shop.flixbus.in/rebooking/login",
                "We cannot alter your booking after it has been confirmed. Please go to our website, select ‘Manage My Booking,’ and provide the needed details to make any changes. Link: https://shop.flixbus.in/rebooking/login"
            ]
        },
        "booking_process": {
            "keywords": ["how to book", "booking process", "book ticket"],
            "responses": [
                "To book a ticket, please visit our website and click on the booking link. Proceed to the checkout page by selecting 'CONTINUE'. Fill out the necessary details: Seat Reservation, Passengers, Contact Information, Payment. Available seat types include Standard free seats, Panorama seats, and Premium seats. Note our gender seating policy ensures female travelers are not seated next to male travelers unless part of the same booking. Carry a valid ID (Aadhar, Passport, or Driving License). Luggage policy allows 7kg hand luggage and 20kg regular luggage free, with additional luggage bookable via Manage My Booking. Payment methods include Credit cards, UPI, and Net Banking. A Rs 5 platform fee applies."
            ]
        },
        "manage_booking_changes": {
            "keywords": ["change date", "change time", "cancel ticket"],
            "responses": [
                "If you wish to change the date or time of your ride, cancel, or postpone it, you can easily make these adjustments through the ‘Manage My Booking’ section. Simply enter your booking number and phone number, click on ‘Retrieve Booking,’ and you will see the options to modify your details. Link: https://shop.flixbus.in/rebooking/login",
                "If you need to reschedule, cancel, or postpone your ride, you can manage these changes through the ‘Manage My Booking’ portal. Enter your booking number and phone number, then click ‘Retrieve Booking’ to find the options for updating your ride details. Link: https://shop.flixbus.in/rebooking/login"
            ]
        },
        "complaint_feedback": {
            "keywords": ["complain", "feedback", "review"],
            "responses": [
                "Thank you so much! We're thrilled to hear that you enjoyed your experience with us. We strive to provide excellent service, and it's always wonderful to receive positive feedback."
            ]
        },
        "rude_behavior": {
            "keywords": ["rude driver", "rude host", "bad behavior"],
            "responses": [
                "I sincerely apologize for the unpleasant experience you had with the driver and the bus host. We deeply regret that their behavior was not up to the standards you expect and deserve. Please be assured that I will escalate this matter to the relevant team for a thorough review and appropriate action. Your feedback is very important to us, and we take such concerns seriously to ensure this doesn’t happen again."
            ]
        },
        "breakdown_refund": {
            "keywords": ["bus breakdown", "ac not working", "refund breakdown"],
            "responses": [
                "Thank you for reaching out, and I sincerely apologize for the inconvenience you've experienced due to the breakdown of the bus. To assist you further and ensure we handle your request appropriately, could you please provide your booking reference number or PNR number, along with the email address or phone number associated with your booking? We are actively working to resolve this and will keep you updated."
            ]
        },
        "route_details": {
            "keywords": ["route details", "bus route"],
            "responses": [
                "I regret to inform you that we don’t have the specific route information about the ride however we have the access for the stop locations associated with your journey with FlixB. These are the stop locations associated with your existing booking. Is there anything else I can help you with?"
            ]
        },
        "change_date": {
            "keywords": ["change date", "reschedule date"],
            "responses": [
                "Yes sure, you can change the date of your journey up to 15 minutes before departure time of the bus via manage my booking section on our website. Link: https://shop.flixbus.in/rebooking/login. After clicking on the above link you can see the option for Booking number and Email or Phone number, then you have to fill those required details and click on the retrieve booking. After that you will be able to change the date of your journey. Please note that the prices are dynamic in nature, and any fare difference will be displayed during the rescheduling process."
            ]
        },
        "route_information": {
            "keywords": ["route information", "bus route info"],
            "responses": [
                "I regret to inform you that we don’t have the route information of the ride however we have the access for the stop location associate with that booking. If you have already booked the ticket and want to know the route information of your ride you may click on the link provided below: https://www.flixbus.in/track/"
            ]
        },
        "flix_lounge": {
            "keywords": ["flix lounge", "anand vihar lounge"],
            "responses": [
                "Thank you for reaching out to us. We apologize for any confusion, but please note that the Flix Lounge facility is not available at the Anand Vihar location. It serves as an operational point for boarding, and only official work takes place there. We suggest waiting at the boarding point for your bus."
            ]
        },
        "bus_delay_less_120": {
            "keywords": ["bus delay less than 120", "short delay"],
            "responses": [
                "I’m really sorry for the delay of the bus. I understand this can be frustrating, and I sincerely apologize for the inconvenience caused. As checked the bus was delayed due to some operational reasons and traffic issues. I have checked that current status, and while the bus is delayed, it is not delayed by 120 minutes or more from your boarding point. According to our T&C, we can only offer a refund if the bus is delayed by more than 120 minutes from your boarding time. Link: https://www.flixbus.in/terms-and-conditions-of-carriage"
            ]
        },
        "bus_delay_over_120": {
            "keywords": ["bus delay over 120", "long delay"],
            "responses": [
                "I sincerely apologize for the delay. I understand how frustrating this can be and I regret the inconvenience caused. Upon investigation, I can confirm that the bus is delayed by more than 2 hours from your boarding point due to operational reasons. If you prefer not to wait, I can proceed with cancelling your ticket and initiate a full refund for you. Would you like me to go ahead with that?"
            ]
        },
        "bus_breakdown_ac": {
            "keywords": ["ac not working", "bus breakdown ac"],
            "responses": [
                "I sincerely apologize for the inconvenience caused due to the bus breakdown and as the AC not working. I understand how uncomfortable this must be for you, and I'm truly sorry. To assist you further, could you please share your booking reference number along with the email address or phone number used during the booking? I’ve already highlighted this issue to our team, and they are working on resolving it as soon as possible."
            ]
        },
        "luggage_policy": {
            "keywords": ["luggage policy", "baggage rules"],
            "responses": [
                "Thank you for reaching out regarding our luggage policy. I’m happy to inform you that you are allowed to bring 7kg of hand luggage and 20kg of regular luggage completely free of charge. Additionally, you may bring one extra luggage item of 20kg per passenger. You can book additional luggage via Manage My Booking: https://shop.flixbus.in/rebooking/login. For more details, please visit: https://www.flixbus.in/service/luggage."
            ]
        },
        "cancel_ticket": {
            "keywords": ["cancel ticket", "ticket cancellation"],
            "responses": [
                "I would like to inform you that you may cancel your ticket from your end up to 15 minutes before the departure time of the bus. You can cancel it through our website via Manage My Booking. Link: https://shop.flixbus.in/rebooking/login. After clicking on this link you can see both the option for booking number and email and phone number, then you can fill with the required details and click on the \"Retrieve Booking\" option. Then you will be able to cancel the ticket and choose between a cash refund or a voucher."
            ]
        },
        "stranded_passenger": {
            "keywords": ["stranded", "left behind"],
            "responses": [
                "We’re very sorry to hear about your situation and understand how frustrating this must be. Could you please share your booking reference number, registered email, and phone number so we can look into this for you? Unfortunately, we are unable to offer a refund or arrange an alternative ride in this situation, as per our company policy."
            ]
        },
        "lost_item": {
            "keywords": ["lost item", "left something", "lost and found"],
            "responses": [
                "We’re very sorry to hear that your belongings were left on the bus. We understand how important this is to you. To assist you in recovering your items, may I kindly request you to fill out our Lost and Found form? Our team will investigate the matter and do their best to locate your belongings."
            ]
        },
        "travel_with_pet": {
            "keywords": ["travel with pet", "pet policy"],
            "responses": [
                "Thank you for your inquiry regarding traveling with pets on Flix. Unfortunately, at this time, we are unable to accommodate pets on our buses. This policy is in place to ensure a safe and pleasant experience for everyone on board. For more details, please refer to our official pet policy."
            ]
        },
        "prices_discounts": {
            "keywords": ["price", "discount", "offer"],
            "responses": [
                "The price shown on your ticket is the final price. You do not need to pay any further amount.",
                "I apologize, but currently, there are no offers or discounts available on our website. However, rest assured that our prices are already set to provide the most convenient options.",
                "Please note that our prices are dynamically adjusted based on demand, availability, and other factors to ensure the best possible experience for all our passengers. We recommend booking early to secure the best available price."
            ]
        },
        "blanket_service": {
            "keywords": ["blanket", "blanket service"],
            "responses": [
                "I regret to inform you but as of now we are not providing Blankets on board however we recommend our customers to carry one along with them for their own comfort and warmth. You may refer to this link you will get to know what kind of services Flix provide.",
                "We're pleased to inform you that blankets and water bottles have been provided for your convenience on all rides."
            ]
        },
        "water_bottle_service": {
            "keywords": ["water", "water bottle"],
            "responses": [
                "We regret to inform you that, as of now, we do not offer water bottle services on our Flix buses. We recommend that passengers bring their own water bottles and any other refreshments they might need for their journey."
            ]
        },
        "washroom_service": {
            "keywords": ["washroom", "restroom", "toilet"],
            "responses": [
                "I regret to inform you that as of now Flix not provided the washroom facilities on the bus. However, the bus host will take care of the comfort breaks while taking your journey."
            ]
        },
        "seat_changes": {
            "keywords": ["change seat", "seat change"],
            "responses": [
                "We apologize, but we are unable to change your seat as it is automatically assigned and based on availability.",
                "Unfortunately, we cannot change your seat because seats are auto-assigned and system-generated based on current availability."
            ]
        },
        "shadow_booking": {
            "keywords": ["shadow booking", "payment not found", "booking not found"],
            "responses": [
                "It's sad to hear about the inconvenience you're experiencing. To assist you further, could you please help me with the following details? 1) Passenger's full name 2) The email ID used for booking 3) The phone number associated with the booking 4) A screenshot of the payment transaction. This information will help us locate your booking and ensure everything is sorted out promptly.",
                "I apologize for the inconvenience, but I couldn't locate any booking matching the details provided by you.",
                "I regret to inform you that, I am unable to locate any booking with the provided details."
            ]
        },
        "no_refund_statement": {
            "keywords": ["no refund", "refund denial"],
            "responses": [
                "After thoroughly investigating the incident, we found that the bus arrived at the designated boarding point and other passengers successfully boarded the bus. Unfortunately, due to these circumstances, we are unable to process a refund for your ticket."
            ]
        },
        "refund_processing": {
            "keywords": ["refund status", "refund processing"],
            "responses": [
                "Your ticket has been cancelled as of (DATE). Please note that it will take up to 7 working days for the amount of (AMOUNT) to be credited back to your account. Don’t worry, your funds are secure and will be refunded within the maximum time frame.",
                "We would like to inform you that your ticket has been cancelled on (DATE). The refund amount of (AMOUNT) will be processed and should appear in your account within 7 working days. Please be assured that your money is safe and will be returned within this period."
            ]
        },
        "refund_tat_crossed": {
            "keywords": ["refund not received", "late refund"],
            "responses": [
                "We would like to inform you that the refund has been initiated from our end on [DATE] for the amount of [XXXX]. Please check with your bank regarding the status of this refund. If you do not receive the amount, kindly share your bank statement up to the current date for further assistance."
            ]
        },
        "closing_statement": {
            "keywords": ["goodbye", "bye", "thanks", "done"],
            "responses": [
                "Thank you for contacting Flix. Have a great day!",
                "I’m happy to have assisted you with your inquiry! If you have any other questions or need further assistance, please feel free to reach out. Have a wonderful day!",
                "It was a pleasure assisting you today. If you need further assistance or have any more questions, don't hesitate to contact us again. Have a wonderful day!"
            ]
        },
        "request_feedback": {
            "keywords": ["feedback", "rate conversation", "survey"],
            "responses": [
                "Looking forward for your valuable feedback towards my response, the link or the option will be there right after the chat ends.",
                "We appreciate your feedback and look forward to hearing from you. You’ll find the link or option available once our chat concludes.",
                "Your feedback towards my response is important for me! The link or option will be provided immediately after our conversation ends."
            ]
        }
    }

response_map = load_response_map()

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Non-empty query string")
    companyemail: EmailStr = Field(..., description="Valid company email address")
    companyid: int = Field(..., ge=1, description="Positive company ID")

    @validator('companyid', pre=True)
    def validate_companyid(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            raise ValueError("Company ID cannot be empty")
        try:
            return int(v)
        except (ValueError, TypeError):
            raise ValueError("Company ID must be a valid integer")

class QueryResponse(BaseModel):
    success: bool
    response: str

class HealthResponse(BaseModel):
    status: str

class TrainingStatusResponse(BaseModel):
    is_training: bool
    learned_phrases: int
    last_retrain: Optional[str]
    training_data_size: int

# Database connection
try:
    engine = create_engine(DATABASE_URL, connect_args={"options": "-c default_transaction_read_only=on"})
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    logger.info("Database connection established successfully")
except SQLAlchemyError as e:
    logger.error(f"Database connection failed: {e}")
    raise

# Custom exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=200,
        content={"success": False, "response": "Please fill the required data"}
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 422:
        logger.error(f"HTTP 422 error: {exc.detail}")
        return JSONResponse(
            status_code=200,
            content={"success": False, "response": "Please fill the required data"}
        )
    logger.error(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

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
                    logger.info(f"Table {table_name} in schema {schema}")
                    return schema
            logger.warning(f"Table {table_name} not found in any schema")
            return None
    except SQLAlchemyError as e:
        logger.error(f"Error finding schema for {table_name}: {str(e)}")
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
    """Preprocess text for TF-IDF and keyword matching."""
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop_words]
    return ' '.join(tokens)

def correct_misspelling(query: str, keywords: list) -> str:
    """Correct misspelled words in query using fuzzy matching."""
    tokens = word_tokenize(query.lower())
    corrected_tokens = []
    for token in tokens:
        if token in stop_words or token.isalnum():
            corrected_tokens.append(token)
            continue
        matches = difflib.get_close_matches(token, keywords, n=1, cutoff=0.8)
        corrected_tokens.append(matches[0] if matches else token)
    return ' '.join(corrected_tokens)

def generate_fallback_response(query: str) -> str:
    """Generate a response for out-of-context or empty queries."""
    tokens = word_tokenize(query.lower())
    if not tokens:
        return intent_templates["out_of_context"]
    if any(word in tokens for word in ["cancel", "cancellation"]):
        return intent_templates["cancellation"]
    elif any(word in tokens for word in ["book", "booking", "ticket"]):
        return intent_templates["booking"]
    elif any(word in tokens for word in ["bus", "travel", "trip"]):
        return "Let me check your travel query. Please provide more details."
    elif any(word in tokens for word in ["delay", "late"]):
        return "Sorry for any delays. Please share your booking details for assistance."
    elif any(word in tokens for word in ["math", "calculate", "+", "-", "*", "/"]):
        return "I can help with travel queries, but for math, try a calculator or ask about tickets!"
    else:
        return intent_templates["out_of_context"]

def generate_ai_response(query: str, intent: str, used_responses_set: set, db_statements: list) -> str:
    """Generate a unique AI response using context and existing responses."""
    tokens = [lemmatizer.lemmatize(t) for t in word_tokenize(query.lower()) if t.isalnum()]
    token = tokens[0] if tokens else "your request"
    templates = {
        "greeting": [
            "Hi there! Excited to help with {token}. What's up?",
            "Hello! I'm here for your {query} needs. How can I assist?"
        ],
        "cancellation": [
            "Need to cancel {token}? Here's how we can proceed.",
            "For {token} cancellation, let me guide you through the steps."
        ],
        "booking": [
            "Looking to book {token}? I can help with that!",
            "Let's get your {token} booked. What's the plan?"
        ],
        "delay": [
            "Sorry about the {token} delay. Let me check for you.",
            "Delays with {token}? I'll assist you promptly."
        ],
        "general": [
            "I can assist with {token}. Tell me more!",
            "Let's resolve your {token} query. What's the issue?"
        ]
    }
    # Extract context from database responses
    context_phrases = []
    for stmt in db_statements:
        if any(t in stmt.lower() for t in tokens):
            context_phrases.append(stmt)
    # Generate response
    category_phrases = templates.get(intent, templates["general"])
    for template in category_phrases:
        response = template.format(token=token, query=query)
        if response not in used_responses_set:
            logger.debug(f"Generated AI response: {response}")
            return response
    # If templates are exhausted, combine context
    if context_phrases:
        base = random.choice(context_phrases).strip("<p>").strip("</p>")
        response = f"Based on your {token} query, {base.lower()}"
        if response not in used_responses_set:
            logger.debug(f"Context-based AI response: {response}")
            return response
    # Fallback with randomization
    counter = len(used_responses_set) + 1
    response = f"Let's address your {token} question. How can I assist? (Unique #{counter})"
    logger.debug(f"Fallback AI response: {response}")
    return response

def load_training_data():
    """Load training data from JSON file."""
    global training_data, training_stats
    try:
        with open("training_data.json", "r") as f:
            data = json.load(f)
            training_data = [preprocess_text(str(stmt)) for stmt in data.get("statements", []) if stmt]
            training_stats["training_data_size"] = len(training_data)
            logger.info(f"Loaded {len(training_data)} statements from training_data.json")
    except FileNotFoundError:
        logger.warning("training_data.json not found, initializing with default statements")
        training_data = [preprocess_text(stmt) for stmt in ["cancel ticket", "book ticket", "bus travel"]]
        training_stats["training_data_size"] = len(training_data)
        with open("training_data.json", "w") as f:
            json.dump({"statements": ["cancel ticket", "book ticket", "bus travel"]}, f, indent=4)
    except json.JSONDecodeError:
        logger.error("Invalid JSON in training_data.json, resetting to default")
        training_data = [preprocess_text(stmt) for stmt in ["cancel ticket", "book ticket", "bus travel"]]
        training_stats["training_data_size"] = len(training_data)
        with open("training_data.json", "w") as f:
            json.dump({"statements": ["cancel ticket", "book ticket", "bus travel"]}, f, indent=4)
    if training_data:
        retrain_vectorizer()

def save_training_data():
    """Save training data to JSON file."""
    global training_stats
    try:
        with open("training_data.json", "w") as f:
            json.dump({"statements": training_data}, f, indent=4)
            training_stats["training_data_size"] = len(training_data)
            logger.info("Saved training data to training_data.json")
    except Exception as e:
        logger.error(f"Error saving training_data.json: {e}")

def fetch_all_db_statements():
    """Fetch and cache all statements from database."""
    global db_statements
    if not check_table_exists():
        return []
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
            db_statements = [str(stmt) for stmt in statements if stmt]
            logger.info(f"Cached {len(db_statements)} database statements")
            return db_statements
    except SQLAlchemyError as e:
        logger.warning(f"Error fetching database statements: {e}")
        return []

def check_data_refresh():
    """Refresh training data from database."""
    global training_data, training_vec, training_stats
    new_statements = fetch_all_db_statements()
    new_training_data = [preprocess_text(str(stmt)) for stmt in new_statements if stmt]
    if new_training_data != training_data:
        training_data = new_training_data
        retrain_vectorizer()
        save_training_data()
        training_stats["training_data_size"] = len(training_data)
        logger.info(f"Refreshed training data with {len(training_data)} statements from database")

def retrain_vectorizer():
    """Retrain TF-IDF vectorizer and cache vectorized data."""
    global vectorizer, training_vec, training_stats
    if training_data:
        try:
            vectorizer = TfidfVectorizer()
            training_vec = vectorizer.fit_transform(training_data)
            training_stats["last_retrain"] = datetime.utcnow().isoformat()
            logger.info(f"TF-IDF vectorizer retrained with {len(training_data)} statements")
        except ValueError as e:
            logger.error(f"Error retraining vectorizer: {e}")

def fetch_similar_statements(query: str, session_key: str) -> str:
    """Fetch unused similar statement from database."""
    preprocessed_query = preprocess_text(query)
    if not training_data or not preprocessed_query or training_vec is None:
        logger.debug("No training data or query for similarity search")
        return ""
    try:
        query_vec = vectorizer.transform([preprocessed_query])
        similarities = cosine_similarity(query_vec, training_vec)[0]
        indices = np.argsort(similarities)[::-1]
        for idx in indices:
            if idx < len(db_statements) and similarities[idx] > 0.05:
                response = db_statements[int(idx)]
                if response not in used_responses[session_key]:
                    logger.debug(f"Selected database response: {response}")
                    return response
        logger.debug("No unused database responses found")
        return ""
    except Exception as e:
        logger.error(f"Error in fetch_similar_statements: {e}")
        return ""

def learn_phrase(query: str, intent: str, response: str):
    """Store query-intent-response mapping for training."""
    global training_stats
    preprocessed_query = preprocess_text(query)
    if preprocessed_query not in QueryDesk_AI_Learning:
        QueryDesk_AI_Learning[preprocessed_query] = intent
        training_data.append(preprocessed_query + f" | {response}")
        training_stats["learned_phrases"] += 1
        training_stats["training_data_size"] = len(training_data)
        retrain_vectorizer()
        save_training_data()
        logger.info(f"Learned new phrase: {query} -> {intent} with response: {response} (Total phrases: {training_stats['learned_phrases']})")

def generate_dynamic_response(query: str, intent: str, companyemail: str, companyid: int) -> Dict[str, bool | str]:
    """Generate non-repeating response: database -> response_map -> AI."""
    session_key = f"{companyemail}_{companyid}"
    corrected_query = correct_misspelling(query, sum([data["keywords"] for data in response_map.values()], []))
    preprocessed_query = preprocess_text(corrected_query)
    if not preprocessed_query:
        logger.debug("Empty preprocessed query, using fallback")
        return {"success": False, "response": generate_fallback_response(query)}
    
    # 1. Try database statements
    similar_statement = fetch_similar_statements(corrected_query, session_key)
    if similar_statement:
        used_responses[session_key].add(similar_statement)
        QueryDesk_AI_Response[query].append({
            "response": similar_statement,
            "companyemail": companyemail,
            "companyid": companyid,
            "timestamp": datetime.utcnow().isoformat()
        })
        learn_phrase(corrected_query, intent, similar_statement)
        return {"success": True, "response": similar_statement}
    
    # 2. Try response_map
    for category, data in response_map.items():
        for keyword in data["keywords"]:
            if keyword in preprocessed_query:
                available_responses = [resp for resp in data["responses"] if resp not in used_responses[session_key]]
                if available_responses:
                    response = random.choice(available_responses)
                    used_responses[session_key].add(response)
                    QueryDesk_AI_Response[query].append({
                        "response": response,
                        "companyemail": companyemail,
                        "companyid": companyid,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    learn_phrase(corrected_query, category, response)
                    return {"success": True, "response": response}
    
    # 3. Try query_response_map
    if corrected_query.lower() in query_response_map:
        response = query_response_map[corrected_query.lower()]
        if response not in used_responses[session_key]:
            used_responses[session_key].add(response)
            QueryDesk_AI_Response[query].append({
                "response": response,
                "companyemail": companyemail,
                "companyid": companyid,
                "timestamp": datetime.utcnow().isoformat()
            })
            learn_phrase(corrected_query, intent, response)
            return {"success": True, "response": response}
    
    # 4. Generate AI response
    response = generate_ai_response(corrected_query, intent, used_responses[session_key], db_statements)
    used_responses[session_key].add(response)
    QueryDesk_AI_Response[query].append({
        "response": response,
        "companyemail": companyemail,
        "companyid": companyid,
        "timestamp": datetime.utcnow().isoformat()
    })
    learn_phrase(corrected_query, intent, response)
    return {"success": True, "response": response}

async def periodic_refresh():
    """Refresh database every 60 seconds."""
    while True:
        check_data_refresh()
        logger.info("Database refresh completed")
        await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_training_data()
    check_data_refresh()
    task = asyncio.create_task(periodic_refresh())
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
    logger.info(f"Received query: {request.query}, companyemail: {request.companyemail}, companyid: {request.companyid}")
    try:
        intent = QueryDesk_AI_Learning.get(preprocess_text(request.query), "general")
        result = generate_dynamic_response(request.query, intent, request.companyemail, request.companyid)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return QueryResponse(success=False, response="Error processing query")

@app.get("/training-status", response_model=TrainingStatusResponse)
async def training_status():
    """Return current training status of the AI model."""
    return TrainingStatusResponse(
        is_training=training_stats["last_retrain"] is not None,
        learned_phrases=training_stats["learned_phrases"],
        last_retrain=training_stats["last_retrain"],
        training_data_size=training_stats["training_data_size"]
    )

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
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))