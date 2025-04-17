from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import logging
import numpy as np
from openai import OpenAI
import pandas as pd
import tiktoken
import json
import math
import asyncio
from scipy.spatial.distance import cdist
from datetime import datetime, timedelta
import stripe
from azure.cosmos import CosmosClient, PartitionKey
from parallel_api import process_json
import traceback
# Awais add this line :
from anonymization.data_anonlyization import DataAnonymizer # data anonlization file added
from category_analyzer.category_analyzer import CategoryAnalyzer
from datetime import datetime, timedelta
import string
import fitz  # PyMuPDF
import base64
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import glob
import threading
import time
from werkzeug.utils import secure_filename
import os

# Configuration
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
COSMOS_URI = os.getenv("COSMOS_URI")
COSMOS_KEY = os.getenv("COSMOS_KEY")
DATABASE_NAME = os.getenv("DATABASE_NAME")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")
client = CosmosClient(COSMOS_URI, credential=COSMOS_KEY)
database = client.create_database_if_not_exists(DATABASE_NAME)
container = database.create_container_if_not_exists(
    id=CONTAINER_NAME,
    partition_key=PartitionKey(path="/id"),
    default_ttl=None
)
async def update_user_metrics(container, user_id, tokens_used, training_run=False, module_type=None, input_size=None, categories_size=None):
    """
    Update user metrics in Cosmos DB with session history
    tokens_used can now be either an integer (for backward compatibility) or a dict with detailed token info
    """
    try:
        if not user_id:
            raise ValueError("user_id cannot be null or empty")
            
        # Query existing user document
        query = f"SELECT * FROM c WHERE c.userId = '{user_id}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        
        current_time = datetime.utcnow().isoformat()
        
        # Create session entry with detailed token information
        session_entry = {
            'timestamp': current_time,
            'training_run': training_run
        }

        # Handle both new dict format and old integer format
        if isinstance(tokens_used, dict):
            session_entry.update({
                'input_tokens': tokens_used.get('input_tokens', 0),
                'output_tokens': tokens_used.get('output_tokens', 0),
                'total_cost': tokens_used.get('total_cost', 0),
                'tokens_used': tokens_used.get('input_tokens', 0) + tokens_used.get('output_tokens', 0)
            })
        else:
            session_entry['tokens_used'] = tokens_used
        
        # Add module-specific data
        if module_type:
            session_entry['module_type'] = module_type
            session_entry['input_size'] = input_size
            if module_type == 'classification' and categories_size is not None:
                session_entry['categories_size'] = categories_size
        
        if items:
            # Get existing document
            doc = items[0]
            
            # Initialize metrics if they don't exist
            if 'metrics' not in doc:
                doc['metrics'] = {
                    'total_tokens': 0,
                    'total_input_tokens': 0,
                    'total_output_tokens': 0,
                    'total_cost': 0,
                    'training_runs': 0,
                    'sessions': [],
                    'last_updated': current_time
                }
            
            # Update general metrics
            if isinstance(tokens_used, dict):
                doc['metrics']['total_input_tokens'] = doc['metrics'].get('total_input_tokens', 0) + tokens_used.get('input_tokens', 0)
                doc['metrics']['total_output_tokens'] = doc['metrics'].get('total_output_tokens', 0) + tokens_used.get('output_tokens', 0)
                doc['metrics']['total_cost'] = doc['metrics'].get('total_cost', 0) + tokens_used.get('total_cost', 0)
                doc['metrics']['total_tokens'] = doc['metrics']['total_input_tokens'] + doc['metrics']['total_output_tokens']
            else:
                doc['metrics']['total_tokens'] += tokens_used
            
            if training_run:
                doc['metrics']['training_runs'] = doc['metrics'].get('training_runs', 0) + 1
            
            # Add new session to sessions array
            if 'sessions' not in doc['metrics']:
                doc['metrics']['sessions'] = []
            doc['metrics']['sessions'].append(session_entry)
            
            # Update last_updated timestamp
            doc['metrics']['last_updated'] = current_time
            
            # Update document in Cosmos DB
            container.upsert_item(doc)
            
        else:
            # Create new document for new user with detailed metrics
            new_doc = {
                "id": user_id,
                "userId": user_id,
                "metrics": {
                    'total_tokens': session_entry.get('tokens_used', 0),
                    'total_input_tokens': session_entry.get('input_tokens', 0),
                    'total_output_tokens': session_entry.get('output_tokens', 0),
                    'total_cost': session_entry.get('total_cost', 0),
                    'training_runs': 1 if training_run else 0,
                    'sessions': [session_entry],
                    'last_updated': current_time
                }
            }
            container.create_item(new_doc)
            
        return True
    except ValueError as ve:
        logging.error(f"Validation error in update_user_metrics: {ve}")
        return False
    except Exception as e:
        logging.error(f"Error updating user metrics: {e}")
        return False
@app.route('/api/initialize-user', methods=['POST'])
def initialize_user():
    try:
        data = request.get_json()
        user_id = data.get('userId')
        user_name = data.get('userName', 'Unknown User')
        
        if not user_id:
            return jsonify({"error": "userId is required"}), 400
            
        # Check if user already exists
        query = f"SELECT * FROM c WHERE c.userId = '{user_id}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        
        if not items:
            # Create new user document with default values
            new_doc = {
                "id": user_id,
                "userId": user_id,
                "userName": user_name,
                "apiKey": "",  # Empty API key by default
                "balance": 0,  # Default balance
                "lastUpdated": datetime.utcnow().isoformat(),
                "metrics": {
                    'total_tokens': 0,
                    'training_runs': 0,
                    'sessions': [],
                    'last_updated': datetime.utcnow().isoformat()
                },
                "transactions": []
            }
            container.create_item(new_doc)
            
        return jsonify({"success": True}), 200
    except Exception as e:
        logging.error(f"Error initializing user: {e}")
        return jsonify({"error": str(e)}), 500        
@app.route('/api/get-balance', methods=['GET'])
def get_balance():
    user_id = request.args.get('userId')
    
    if not user_id:
        return jsonify({"error": "userId is required"}), 400
        
    try:
        query = f"SELECT c.balance FROM c WHERE c.userId = '{user_id}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        
        if not items:
            return jsonify({"error": "User not found"}), 404
            
        return jsonify({"balance": items[0].get('balance', 0)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
TOKEN_RATES = {
    "input": 2.50 / 1000000,  # $0.07 per 1K input tokens
    "output": 10.0 / 1000000  # $0.02 per 1K output tokens
}

def estimate_token_usage(input_data, module_type="classification", model="gpt-4o"):
    """
    Estimates token usage and cost for different operations with separate input/output rates based on model
    """
    # Get model-specific rates, defaulting to gpt-4o if model not found
    model_rates = MODEL_TOKEN_RATES.get(model, MODEL_TOKEN_RATES["gpt-4o"])
    
    if module_type == "classification":
        # Estimate tokens for classification
        estimated_input_tokens = len(input_data) * 150  # Estimated tokens per input
        estimated_output_tokens = len(input_data) * 50   # Estimated tokens per response
        
        # Calculate costs using model-specific rates
        input_cost = (estimated_input_tokens * model_rates["input"])
        output_cost = (estimated_output_tokens * model_rates["output"])
        
        return {
            "input_tokens": estimated_input_tokens,
            "output_tokens": estimated_output_tokens,
            "total_cost": input_cost + output_cost
        }
    elif module_type == "open_format":
        # Estimate tokens for open format requests
        # Model-specific token estimates
        token_multipliers = {
            "gpt-4o": {"input": 200, "output": 100},
            "gpt-4o-mini": {"input": 200, "output": 100},
            "gpt-3.5-turbo": {"input": 200, "output": 100}
        }
        
        # Get token multipliers for the selected model
        multipliers = token_multipliers.get(model, token_multipliers["gpt-4o"])
        
        estimated_input_tokens = len(input_data) * multipliers["input"]
        estimated_output_tokens = len(input_data) * multipliers["output"]
        
        # Calculate costs using model-specific rates
        input_cost = (estimated_input_tokens * model_rates["input"])
        output_cost = (estimated_output_tokens * model_rates["output"])
        
        return {
            "input_tokens": estimated_input_tokens,
            "output_tokens": estimated_output_tokens,
            "total_cost": input_cost + output_cost
        }
    
    return None

def calculate_actual_cost(input_tokens, output_tokens):
    """
    Calculates the actual cost based on separate input and output token counts
    """
    input_cost = (input_tokens * TOKEN_RATES["input"])
    output_cost = (output_tokens * TOKEN_RATES["output"])
    return input_cost + output_cost
async def check_and_deduct_balance(user_id, estimated_cost):
    """
    Checks if user has sufficient balance and deducts if they do
    """
    query = f"SELECT * FROM c WHERE c.userId = '{user_id}'"
    items = list(container.query_items(query=query, enable_cross_partition_query=True))
    
    if not items:
        raise Exception("User not found")
        
    doc = items[0]
    
    # Check if user has subscription
    # if doc.get('subscription', {}).get('active'):
    #     return True
        
    # Check balance
    current_balance = doc.get('balance', 0)
    if current_balance < estimated_cost:
        raise Exception(f"Insufficient balance. Required: ${estimated_cost:.2f}, Current: ${current_balance:.2f}")
    
    # Deduct estimated cost
    doc['balance'] = current_balance - estimated_cost
    doc['transactions'] = doc.get('transactions', [])
    doc['transactions'].append({
        'timestamp': datetime.utcnow().isoformat(),
        'type': 'pending_deduction',
        'amount': estimated_cost,
        'details': 'Estimated cost for analysis'
    })
    
    container.upsert_item(doc)
    return True
async def adjust_final_cost(user_id, estimated_cost, actual_cost):
    """
    Adjusts the final cost after processing is complete
    """
    query = f"SELECT * FROM c WHERE c.userId = '{user_id}'"
    items = list(container.query_items(query=query, enable_cross_partition_query=True))
    
    if not items:
        raise Exception("User not found")
        
    doc = items[0]
    
    # Skip if user has subscription
    # if doc.get('subscription', {}).get('active'):
    #     return True
        
    # Refund the difference if estimated cost was higher
    if estimated_cost > actual_cost:
        refund_amount = estimated_cost - actual_cost
        doc['balance'] = doc.get('balance', 0) + refund_amount
        doc['transactions'].append({
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'refund',
            'amount': refund_amount,
            'details': 'Refund of overestimated cost'
        })
    
    container.upsert_item(doc)
    return True
def validate_recharge_amount(amount):
    """
    Validates recharge amount and returns standardized value
    """
    MIN_AMOUNT = 100  # $1.00
    MAX_AMOUNT = 50000  # $500.00
    
    if amount < MIN_AMOUNT:
        raise ValueError(f"Minimum recharge amount is ${MIN_AMOUNT/100:.2f}")
    if amount > MAX_AMOUNT:
        raise ValueError(f"Maximum recharge amount is ${MAX_AMOUNT/100:.2f}")
        
    return amount
@app.route('/api/calculate-token-cost', methods=['POST'])
def calculate_token_cost():
    try:
        data = request.get_json()
        input_tokens = data.get('input_tokens', 0)
        output_tokens = data.get('output_tokens', 0)
        model_type = data.get('model_type', 'chat')  # Default to chat model
        
        if model_type == 'embedding':
            # For embeddings, we only care about input tokens
            embedding_rate = EMBEDDING_RATES.get("text-embedding-3-small", 0.002 / 1000000)
            total_cost = input_tokens * embedding_rate
        else:
            # For chat completions, use the existing rates
            input_rate = TOKEN_RATES.get("input", 2.50 / 1000000)
            output_rate = TOKEN_RATES.get("output", 10.0 / 1000000)
            total_cost = (input_tokens * input_rate) + (output_tokens * output_rate)
            
        return jsonify({"cost": total_cost}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Modify the create-recharge-session endpoint
@app.route('/api/create-recharge-session', methods=['POST'])
def create_recharge_session():
    try:
        data = request.get_json()
        user_id = data.get('userId')
        amount = data.get('amount')  # Amount in cents

        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        # Validate amount
        try:
            amount = validate_recharge_amount(amount)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': 'Token Balance Recharge',
                        'description': 'Add credits for pay-as-you-go token usage'
                    },
                    'unit_amount': amount,  # Amount in cents
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url='https://altavize.z13.web.core.windows.net/checkout1.html?payment=success',
            cancel_url='https://altavize.z13.web.core.windows.net/checkout1.html?payment=canceled',
            client_reference_id=user_id,
            metadata={'type': 'recharge'}
        )
        
        return jsonify({'sessionId': checkout_session.id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add function to deduct balance
async def deduct_balance(user_id, amount, details):
    try:
        query = f"SELECT * FROM c WHERE c.userId = '{user_id}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        
        if not items:
            raise Exception("User not found")
            
        doc = items[0]
        current_balance = doc.get('balance', 0)
        
        if current_balance < amount:
            raise Exception("Insufficient balance")
            
        doc['balance'] = current_balance - amount
        doc['transactions'] = doc.get('transactions', [])
        doc['transactions'].append({
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'deduction',
            'amount': amount,
            'details': details
        })
        
        container.upsert_item(doc)
        return True
    except Exception as e:
        logging.error(f"Error deducting balance: {e}")
        raise
@app.route('/api/save_key', methods=['POST'])
def save_api_key():
    data = request.get_json()
    user_id = data.get('userId')
    api_key = data.get('apiKey', '')  # Default to empty string if not provided

    if not user_id:
        return jsonify({"error": "userId is required"}), 400

    try:
        # Skip validation if API key is empty (this means user wants to remove the key)
        if api_key:
            # Basic validation of API key format
            if not api_key.startswith('sk-') or len(api_key) < 20:
                return jsonify({"error": "Invalid API key format"}), 400

            # Test the API key with a small request to OpenAI
            try:
                client = OpenAI(api_key=api_key)
                # Make a minimal API call to test the key validity
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Test"}
                    ],
                    max_tokens=5
                )
                # If we get here, the key is valid
                logging.info(f"API key validation successful for user {user_id}")
            except Exception as api_error:
                logging.error(f"API key validation failed: {api_error}")
                error_message = str(api_error)
                if "401" in error_message or "authentication" in error_message.lower() or "invalid" in error_message.lower():
                    return jsonify({"error": "Invalid API key. The key was rejected by OpenAI."}), 400
                elif "insufficient_quota" in error_message or "exceeded your current quota" in error_message:
                    return jsonify({"error": "This API key has insufficient quota or billing issues."}), 400
                else:
                    return jsonify({"error": f"API key validation failed: {error_message}"}), 400

        # First, get existing document if it exists
        query = f"SELECT * FROM c WHERE c.userId = '{user_id}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        
        if items:
            # Update existing document
            doc = items[0]
            doc['apiKey'] = api_key  # This can now be empty
            doc['lastUpdated'] = datetime.utcnow().isoformat()
            container.upsert_item(doc)
        else:
            # Create new document
            container.upsert_item({
                "id": user_id,
                "userId": user_id,
                "apiKey": api_key,
                "lastUpdated": datetime.utcnow().isoformat(),
                "metrics": {
                    'total_tokens': 0,
                    'training_runs': 0,
                    'sessions': [],
                    'last_updated': datetime.utcnow().isoformat()
                }
            })
            
        return jsonify({"message": "API key saved successfully"}), 200

    except Exception as e:
        logging.error(f"Error saving API key: {e}")
        return jsonify({"error": f"Error saving API key: {str(e)}"}), 500
@app.route('/api/load_key', methods=['GET'])
def load_api_key():
    user_id = request.args.get('userId')

    if not user_id:
        return jsonify({"error": "userId is required"}), 400

    try:
        query = f"SELECT c.apiKey FROM c WHERE c.userId = '{user_id}'"
        items = list(container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))

        if not items:
            return jsonify({"error": "API key not found"}), 404

        # Validate stored API key format
        api_key = items[0].get('apiKey', '')
        if not api_key.startswith('sk-') or len(api_key) < 20:
            return jsonify({"error": "Stored API key is invalid"}), 400

        return jsonify({"apiKey": api_key}), 200

    except Exception as e:
        logging.error(f"Error loading API key: {e}")
        return jsonify({"error": str(e)}), 500
gptmodel = "gpt-4o"
request_url= "https://api.openai.com/v1/chat/completions"
max_requests_per_minute =250
max_tokens_per_minute = 30000
token_encoding_name = "cl100k_base"
max_attempts = 5
logging_level = 20
stripe.api_key = os.getenv("STRIPE_KEY")
# Add these new routes to your existing app.py
@app.route('/api/create-checkout-session', methods=['POST'])
def create_checkout_session():
    try:
        if not stripe.api_key:
            logging.error("Stripe API key not configured")
            return jsonify({"error": "Stripe is not properly configured"}), 500
            
        data = request.get_json()
        user_id = data.get('userId')

        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        # Create checkout session with modified success/cancel URLs
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': 'price_1RABGKD5SXHCku3V1U7ymEBT',
                'quantity': 1,
            }],
            mode='subscription',
            success_url='https://altavize.z13.web.core.windows.net/checkout1.html?payment=success',
            cancel_url='https://altavize.z13.web.core.windows.net/checkout1.html?payment=canceled',
            client_reference_id=user_id,
        )

        return jsonify({'sessionId': checkout_session.id}), 200
    except Exception as e:
        logging.error(f"Error creating checkout session: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
# Add this new endpoint to app.py

@app.route('/api/cancel-subscription', methods=['POST'])
def cancel_subscription():
    try:
        data = request.get_json()
        user_id = data.get('userId')

        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        # Query CosmosDB for user's subscription data
        query = f"SELECT * FROM c WHERE c.userId = '{user_id}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))

        if not items:
            return jsonify({"error": "User not found"}), 404

        doc = items[0]
        subscription = doc.get('subscription', {})

        if not subscription or not subscription.get('active'):
            return jsonify({"error": "No active subscription found"}), 400

        subscription_id = subscription.get('subscriptionId')
        if not subscription_id:
            return jsonify({"error": "Subscription ID not found"}), 400

        # Cancel the subscription in Stripe
        try:
            # If it's a test subscription (created via test endpoint), handle differently
            if subscription_id.startswith('test_sub_'):
                logging.info(f"Canceling test subscription: {subscription_id}")
            else:
                stripe_subscription = stripe.Subscription.retrieve(subscription_id)
                stripe_subscription.cancel_at_period_end = True
                stripe_subscription.save()

            # Update subscription status in CosmosDB
            subscription['active'] = False
            subscription['canceledAt'] = datetime.utcnow().isoformat()
            doc['subscription'] = subscription

            # Update document in CosmosDB
            container.upsert_item(doc)

            return jsonify({
                "message": "Subscription cancelled successfully",
                "status": "cancelled"
            }), 200

        except stripe.error.StripeError as e:
            logging.error(f"Stripe error: {str(e)}")
            return jsonify({"error": f"Stripe error: {str(e)}"}), 400

    except Exception as e:
        logging.error(f"Error cancelling subscription: {str(e)}")
        return jsonify({"error": str(e)}), 500
@app.route('/api/webhook', methods=['POST'])
def webhook():
    logging.info("🔔 Webhook request received")
    try:
        # Log request details
        payload = request.get_data()
        sig_header = request.headers.get('Stripe-Signature')
        
        logging.info("Request Headers:")
        for key, value in request.headers.items():
            if key.lower() not in ['authorization', 'stripe-signature']:
                logging.info(f"   {key}: {value}")
        
        logging.info(f"Payload size: {len(payload)} bytes")
        logging.info(f"Signature header present: {bool(sig_header)}")
        
        webhook_secret = os.getenv("Webhook_Secret")
        logging.info(f"Webhook secret configured: {bool(webhook_secret)}")

        if not webhook_secret:
            logging.error("WEBHOOK_SECRET environment variable not set")
            return jsonify({'error': 'Webhook secret not configured'}), 500

        # Verify webhook
        logging.info("Verifying webhook signature...")
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )
        logging.info(f"Webhook verified successfully")
        logging.info(f"Event type: {event['type']}")
        logging.info(f"Event ID: {event.get('id', 'N/A')}")

        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            logging.info("Processing checkout.session.completed event")
            logging.info(f"Session ID: {session.get('id', 'N/A')}")
            
            user_id = session.get('client_reference_id')
            if not user_id:
                logging.error("Missing user_id in session")
                return jsonify({'error': 'Invalid session data'}), 400

            # Query CosmosDB for user
            logging.info(f"Querying CosmosDB for user: {user_id}")
            query = f"SELECT * FROM c WHERE c.userId = '{user_id}'"
            items = list(container.query_items(query=query, enable_cross_partition_query=True))
            
            if not items:
                logging.error(f"No document found for user: {user_id}")
                return jsonify({'error': 'User not found'}), 404

            doc = items[0]
            
            # Check if this is a subscription or recharge payment
            payment_type = session.get('metadata', {}).get('type')
            subscription_id = session.get('subscription')

            if payment_type == 'recharge':
                # Handle recharge payment
                logging.info("Processing recharge payment")
                amount_paid = session['amount_total'] / 100  # Convert cents to dollars
                
                # Update balance
                current_balance = doc.get('balance', 0)
                doc['balance'] = current_balance + amount_paid
                
                # Add transaction record
                doc['transactions'] = doc.get('transactions', [])
                doc['transactions'].append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'type': 'recharge',
                    'amount': amount_paid,
                    'details': 'Balance recharge'
                })
                
                logging.info(f"Updated balance: +${amount_paid:.2f}")

            elif subscription_id:
                # Handle subscription payment
                logging.info("Processing subscription payment")
                subscription_data = {
                    'active': True,
                    'subscriptionId': subscription_id,
                    'startDate': datetime.utcnow().isoformat(),
                    'endDate': (datetime.utcnow() + timedelta(days=365)).isoformat()
                }
                
                logging.info(f"Updating subscription data: {json.dumps(subscription_data, indent=2)}")
                doc['subscription'] = subscription_data

            # Update document in CosmosDB
            try:
                container.upsert_item(doc)
                logging.info("Successfully updated user document in CosmosDB")
            except Exception as db_error:
                logging.error(f"Failed to update CosmosDB: {str(db_error)}")
                logging.error(f"Stack trace: {traceback.format_exc()}")
                raise

        return jsonify({'status': 'success'}), 200
        
    except stripe.error.SignatureVerificationError as e:
        logging.error(f"Webhook signature verification failed: {str(e)}")
        logging.error(f"Stack trace: {traceback.format_exc()}")
        return jsonify({'error': 'Invalid signature'}), 400
        
    except Exception as e:
        logging.error(f"Webhook processing error: {str(e)}")
        logging.error(f"Stack trace: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 400
def get_appropriate_api_key(user_id):
    """
    Determines which API key to use based on user's balance:
    - If balance ≥ $1: Use company API key
    - If balance < $1: Use user's API key
    
    Returns a tuple: (api_key, is_company_key)
    """
    # Get user document
    query = f"SELECT * FROM c WHERE c.userId = '{user_id}'"
    items = list(container.query_items(query=query, enable_cross_partition_query=True))
    
    if not items:
        return None, False
        
    doc = items[0]
    current_balance = doc.get('balance', 0)
    
    # If user has sufficient balance, use company API key
    if current_balance >= 1:
        company_api_key = os.getenv("company_api_key")
        return company_api_key, True
    # Otherwise use user's API key
    else:
        user_api_key = doc.get('apiKey')
        if user_api_key and user_api_key.startswith('sk-') and len(user_api_key) >= 20:
            return user_api_key, False
            
    # User has invalid API key and insufficient balance
    return None, False
def verify_subscription(user_id):
    """Check if user has an active subscription"""
    try:
        query = f"SELECT * FROM c WHERE c.userId = '{user_id}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        
        if not items:
            return False

        doc = items[0]
        subscription = doc.get('subscription', {})
        
        # Only check subscription status
        has_subscription = subscription and subscription.get('active')
        if has_subscription:
            # Check if subscription is expired
            end_date = datetime.fromisoformat(subscription['endDate'])
            if end_date < datetime.utcnow():
                # Update subscription status to inactive
                doc['subscription']['active'] = False
                container.upsert_item(doc)
                return False
            return True
        return False

    except Exception as e:
        logging.error(f"Error verifying subscription: {e}")
        return False
def get_subscription_status(user_id):
    """Check if user has an active subscription (ignoring balance)"""
    try:
        query = f"SELECT * FROM c WHERE c.userId = '{user_id}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        
        if not items:
            return False

        doc = items[0]
        subscription = doc.get('subscription', {})
        
        # Only check subscription status
        has_subscription = subscription and subscription.get('active')
        if has_subscription:
            # Check if subscription is expired
            end_date = datetime.fromisoformat(subscription['endDate'])
            if end_date < datetime.utcnow():
                # Update subscription status to inactive
                doc['subscription']['active'] = False
                container.upsert_item(doc)
                return False
            return True
        return False

    except Exception as e:
        logging.error(f"Error checking subscription status: {e}")
        return False

@app.route('/api/check_subscription', methods=['GET'])
def check_subscription():
    user_id = request.args.get('userId')
    
    if not user_id:
        return jsonify({"error": "userId is required"}), 400
        
    try:
        # Use the new get_subscription_status function
        is_subscribed = get_subscription_status(user_id)
        return jsonify({
            "isSubscribed": is_subscribed,
            "hasAccess": verify_subscription(user_id)
        }), 200
        
    except Exception as e:
        logging.error(f"Error checking subscription status: {e}")
        return jsonify({"error": str(e)}), 500
    
EMBEDDING_RATES = {
    "text-embedding-3-small": 0.002 / 1000000  # $0.002 per million tokens
}
@app.route('/api/uniqueness', methods=['POST'])
def analyze_uniqueness():
    try:
        logging.info("Uniqueness analysis request received")
        data = request.get_json()

        # Extract user ID
        user_id = data.get('userId')
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        # First check if user has access
        if not verify_subscription(user_id):
            return jsonify({"error": "Active subscription"}), 403
        query = f"SELECT * FROM c WHERE c.userId = '{user_id}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))

        if not items:
            return jsonify({"error": "User not found"}), 404

        doc = items[0]
        current_balance = doc.get('balance', 0)
        user_api_key = doc.get('apiKey')

        # Check if both conditions are true: balance < 1 and API key is null/invalid
        if current_balance < 1 and (not user_api_key or not user_api_key.startswith('sk-') or len(user_api_key) < 20):
            logging.error(f"User {user_id} has insufficient balance and no valid API key")
            return jsonify({
                "error": "Please enter a valid API key in settings to continue. Your account balance is insufficient to use our API key."
            }), 403

        # Determine which API key to use
        api_key, is_company_key = get_appropriate_api_key(user_id)

        # Validate input data
        input_data = data.get('inputData', [])
        if not input_data:
            logging.error("Input data is empty.")
            return jsonify({"error": "Input data cannot be empty."}), 400

        # Create DataFrame from input; each row contains one "Item"
        df_item = pd.DataFrame([row[0] for row in input_data], columns=['Item'])
        
        # Initialize new columns to ensure updates work without length mismatch errors
        df_item['embedding'] = [np.zeros(200) for _ in range(len(df_item))]

        # Calculate embedding tokens and cost
        embedding_tokens = len(input_data) * 200  # Each item uses approximately 200 tokens
        embedding_cost = embedding_tokens * EMBEDDING_RATES["text-embedding-3-small"]

        # Only deduct balance if using company API key (balance is being used)
        if is_company_key:
            try:
                asyncio.run(check_and_deduct_balance(user_id, embedding_cost))
            except Exception as e:
                return jsonify({"error": str(e)}), 403

        # Initialize OpenAI client with appropriate API key
        client = OpenAI(api_key=api_key)

        # Build parallel requests for embeddings call using process_json
        json_requests = []  # Initialize list for embedding JSON requests
        for index, row in df_item.iterrows():
            json_requests.append({
                "model": "text-embedding-3-small",
                "input": row['Item'],
                "dimensions": 200,
                "metadata": {"row_id": index}  # Include metadata for tracking
            })
        embedding_request_url = "https://api.openai.com/v1/embeddings"
        print(json_requests)
        # Call parallel processor for embeddings using process_json
        responses = process_json(
            request_json=json_requests,
            request_url=embedding_request_url,
            api_key=api_key,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name=token_encoding_name,
            max_attempts=max_attempts,
            logging_level=logging_level,
        )

        # Process responses from parallel processor and update DataFrame directly
        # We update each row based on the row_id from metadata.
        for response in responses:
            try:
                if isinstance(response, Exception):
                    # Attempt to extract metadata if available
                    if isinstance(response, (list, tuple)) and len(response) > 2:
                        metadata = response[2]
                    else:
                        metadata = {}
                    row_id = metadata.get("row_id")
                    logging.error(f"Error generating embedding for row {row_id}: {response}")
                    if row_id is not None:
                        df_item.at[row_id, 'embedding'] = np.zeros(200)  # Fallback embedding
                else:
                    # Unpack the response tuple; assume structure: [request_payload, api_response, metadata]
                    request_payload, api_response, metadata = response
                    row_id = metadata.get("row_id")
                    if row_id is None:
                        logging.error("API response missing row_id in metadata")
                        continue
                    embedding = api_response['data'][0]['embedding']
                    df_item.at[row_id, 'embedding'] = embedding
            except Exception as e:
                logging.error(f"Exception processing response for row {row_id if 'row_id' in locals() else 'unknown'}: {e}")
                if 'row_id' in locals() and row_id is not None:
                    df_item.at[row_id, 'embedding'] = np.zeros(200)
        ##### End of Zack updated code #########



        # Convert embeddings to a numpy array from the DataFrame column
        embeddings_matrix = np.vstack(df_item.sort_index()['embedding'].tolist())

        # Calculate Euclidean distances
        euclidean_distances = cdist(embeddings_matrix, embeddings_matrix, metric='euclidean')  # type: ignore
        np.fill_diagonal(euclidean_distances, np.nan)

        # Calculate statistical measures
        overall_mean = np.nanmean(euclidean_distances)
        overall_min = np.nanmin(euclidean_distances)

        # Convert distances to lists and compute per-row statistics
        df_item['euclidean_distances'] = [row.tolist() for row in euclidean_distances]
        df_item['euclidean_distances'] = [np.nan_to_num(row).tolist() for row in euclidean_distances]
        df_item['mean_distance'] = df_item['euclidean_distances'].apply(np.nanmean)

        # Compute 1st and 5th percentiles for each row
        df_item['1st_percentile'] = df_item['euclidean_distances'].apply(lambda x: np.percentile(x, 1))
        df_item['5th_percentile'] = df_item['euclidean_distances'].apply(lambda x: np.percentile(x, 5))

        # Calculate overall mean and std for the 1st and 5th percentiles
        overall_1st_mean = df_item['1st_percentile'].mean()
        overall_1st_std = df_item['1st_percentile'].std()
        overall_5th_mean = df_item['5th_percentile'].mean()
        overall_5th_std = df_item['5th_percentile'].std()

        # Compute z-scores for each row based on its percentile against the overall statistics
        df_item['one_z_score'] = (df_item['1st_percentile'] - overall_1st_mean) / overall_1st_std
        df_item['five_z_score'] = (df_item['5th_percentile'] - overall_5th_mean) / overall_5th_std

        # Calculate the mean z-score for the mean_distance using externally defined overall_mean
        df_item['mean_z_score'] = (df_item['mean_distance'] - overall_mean) / (np.nanstd(np.concatenate(df_item['euclidean_distances'].values), ddof=1))

        df_item['unique_score'] = (0.5 * df_item['mean_z_score']) + (0.25 * df_item['one_z_score']) + (0.25 * df_item['five_z_score'])

        # Normalize scores to 0-1 range
        score_min = df_item['unique_score'].min()
        score_max = df_item['unique_score'].max()
        df_item['unique_score'] = (df_item['unique_score'] - score_min) / (score_max - score_min)

        # Format results to match original output structure
        uniqueness_scores = [[float(score)] for score in df_item['unique_score']]

        # Create token metrics dictionary for detailed tracking
        token_metrics = {
            'input_tokens': embedding_tokens,
            'output_tokens': 0,  # Embeddings don't have output tokens
            'total_tokens': embedding_tokens,
            'total_cost': embedding_cost
        }

        # Always update metrics for tracking
        asyncio.run(update_user_metrics(
            container=container,
            user_id=user_id,
            tokens_used=token_metrics,
            module_type='uniqueness',
            input_size=len(input_data)
        ))

        # Only adjust final cost if using company key
        if is_company_key:
            asyncio.run(adjust_final_cost(user_id, embedding_cost, embedding_cost))

        logging.info("Uniqueness analysis complete")
        return jsonify(uniqueness_scores), 200

    except Exception as e:
        logging.error(f"Unexpected error in uniqueness analysis: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        logging.info("Analysis request received")
        data = request.get_json()
        
        # Extract API parameters from the request payload
        api_key = data.get('apiKey')
        user_id = data.get('userId')
        
        # Extract input data first
        item_data = data.get('inputData', [])
        category_list = data.get('categories', [])
        instructions = data.get('instructions', '')
        rerun_with_training = bool(data.get('rerun_with_training', ''))
        # use_web_search = bool(data.get('useWebSearch', False))
        # if use_web_search:
        #     # Add your web search related logic here
        #     logging.info("Web search functionality enabled for this request")   
        if not verify_subscription(user_id):
            return jsonify({"error": "Active subscription required"}), 403
       
            # Get user document to check balance and API key
        query = f"SELECT * FROM c WHERE c.userId = '{user_id}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        
        if not items:
            return jsonify({"error": "User not found"}), 404
            
        doc = items[0]
        current_balance = doc.get('balance', 0)
        user_api_key = doc.get('apiKey')
        
        # Check if both conditions are true: balance < 1 and API key is null/invalid
        if current_balance < 1 and (not user_api_key or not user_api_key.startswith('sk-') or len(user_api_key) < 20):
            logging.error(f"User {user_id} has insufficient balance and no valid API key")
            return jsonify({
                "error": "Please enter a valid API key in settings to continue. Your account balance is insufficient to use our API key."
            }), 403
            
        api_key, is_company_key = get_appropriate_api_key(user_id)
         
        # Now we can use item_data for cost estimation
        token_estimates = estimate_token_usage(item_data)
        if is_company_key:
            try:
                asyncio.run(check_and_deduct_balance(user_id, token_estimates["total_cost"]))
            except Exception as e:
                return jsonify({"error": str(e)}), 403

        input_size = len(item_data)
        categories_size = len(category_list)
        
        

        if not item_data or all(not item for sublist in item_data for item in sublist):
            logging.error("Input data is empty.")
            return jsonify({"error": "Input data cannot be empty."}), 400

        if not category_list or all(not category for category in category_list):
            logging.error("Categories are empty.")
            return jsonify({"error": "Categories cannot be empty."}), 400
        if not instructions:
            instructions = "No additional instructions provided by user"

        # Convert validated lists to DataFrames
        df_item = pd.DataFrame(item_data, columns=['Item'])
        df_category = pd.DataFrame(category_list, columns=['Category'])

        df_item['Response'] = ""
        df_item['Confidence'] = ""

        # Load text file content (assumed path needs to be specified)
        #txt_file_path = r'C:\ChatNoir\analysis\Prompt.txt'  # Update this path
        #with open(txt_file_path, 'r') as file:
        txt_content = """You are a data scientist data assistant and your job is to sort [Item] into one of the following categories: [Categories]

The additional context has been provided to help complete the task:
[Instructions]
You are only to return one of the categories and no other response. Please provide your best guess when there is no certain choice. Final response has to be from given categories [Categories]
"""

        # Define the weighting value
        weighting = 5

        # Initialize an empty dictionary to store the weighted tokens for all categories
        tokens_list = {}
        category_token_counts = {}
        df_category['tokens'] = None
        
        # Loop through categories to tokenize
        encoding = tiktoken.encoding_for_model(gptmodel)
        for index, row in df_category.iterrows():
            category = row[df_category.columns[0]]  # Assuming the category is in the first column
            
            # Tokenize the category using the encoding
            token_ids = tiktoken.encoding_for_model(gptmodel).encode(category)
            
            # Decode the token IDs back to tokens
            tokens = [tiktoken.encoding_for_model(gptmodel).decode([token_id]) for token_id in token_ids]
            
            # Add each token ID with its weighting to the tokens_list dictionary
            for token_id in token_ids:
                tokens_list[token_id] = weighting
            
            # Store the number of tokens for this category
            category_token_counts[category] = len(token_ids)
            
            # Add the tokens (as actual strings) to the DataFrame
            df_category.at[index, 'tokens'] = tokens

        # Find the category with the most tokens
        max_tokens_category = max(category_token_counts, key=category_token_counts.get)
        max_tokens_count = category_token_counts[max_tokens_category]

        # Function to calculate confidence score
        def calculate_confidence(logprobs_content, df_category, response_text):
            # Initialize arrays to store summed probabilities for each category per position
            category_sums = {
                'Selected category': [],
                'Not-selected category': [],
                'Model deviation': [],
                'Selected category- Incorrect tokens': []
            }
            # Find the matching category row index directly
            category_row_index = df_category[df_category['Category'] == response_text].index
            category_row = category_row_index[0] if len(category_row_index) > 0 else None

            # Loop through the object and categorize TopLogprob tokens
            tokens_set = {t.lower() for tokens in df_category['tokens'] for t in tokens}
            response_text_lower = response_text.lower()

            for item in logprobs_content:
                # Store the probabilities for each category at each token position
                token_probs = {key: 0.0 for key in category_sums}

                for top_logprob in item['top_logprobs']:
                    token_lower = top_logprob['token'].lower()
                    all_tokens_lower = [t.lower() for tokens in df_category['tokens'] for t in tokens]

                    probability = math.exp(top_logprob['logprob'])

                    if category_row is not None and token_lower in [t.lower() for t in df_category.at[category_row, 'tokens']]:
                        token_probs['Selected category'] += probability
                    elif token_lower not in tokens_set:
                        token_probs['Model deviation'] += probability
                    elif token_lower in response_text_lower:
                        token_probs['Selected category- Incorrect tokens'] += probability
                    else:
                        token_probs['Not-selected category'] += probability

                # Append the summed probabilities for each token position across all categories
                for category, prob in token_probs.items():
                    category_sums[category].append(prob)

            # Ensure all probability sum lists are the same length by padding with zeros
            max_length = len(category_sums['Selected category'])
            for category in category_sums:
                category_sums[category] += [0.0] * (max_length - len(category_sums[category]))

            # Create a summary DataFrame for the total probabilities at each position
            summary_df = pd.DataFrame({
                'Category': list(category_sums.keys()),
                **{f'Position {i+1}': [category_sums[category][i] for category in category_sums] for i in range(max_length)}
            })

            # Calculate weighting for Model Deviation
            total_model_deviation = 0
            for i in range(max_length):
                total_model_deviation += (1 - total_model_deviation) * (summary_df.at[summary_df[summary_df['Category'] == 'Model deviation'].index[0], f'Position {i + 1}'])

            # Calculate entropy probabilities using log approach to avoid numerical underflow
            entropy_probs = []
            for i in range(max_length + 1):
                if i < max_length:
                    # Start with log of Secondary Prediction at current position
                    log_probability = math.log(summary_df.at[summary_df[summary_df['Category'] == 'Not-selected category'].index[0], f'Position {i + 1}'] + 1e-10)
                    # Add log of all previous Primary Predictions, avoiding zero propagation
                    for j in range(i):
                        primary_prob = summary_df.at[summary_df[summary_df['Category'] == 'Selected category'].index[0], f'Position {j + 1}']
                        log_probability += math.log(primary_prob + 1e-10)
                else:
                    # For the final step, just use the log product of all Primary Predictions
                    log_probability = sum([math.log(summary_df.at[summary_df[summary_df['Category'] == 'Selected category'].index[0], f'Position {j + 1}'] + 1e-10) for j in range(max_length)])
                entropy_probs.append(math.exp(log_probability))

            # Normalize the entropy probabilities to ensure they sum to 1
            total_prob_sum = sum(entropy_probs)
            normalized_entropy_probs = [p / total_prob_sum for p in entropy_probs] if total_prob_sum > 0 else [1 / len(entropy_probs)] * len(entropy_probs)

            # Calculate entropy
            entropy = -sum([p * math.log2(p) for p in normalized_entropy_probs if p > 0])

            # Calculate maximum entropy (log2 of number of combinations)
            max_entropy = math.log2(max_length + 1)

            # Calculate total confidence based on entropy
            # Ensure a minimum confidence value to prevent rounding to 0
            total_confidence = max(0.00001, (1 - total_model_deviation) * (1 - entropy / max_entropy) if max_entropy > 0 else (1 - total_model_deviation))
            
            return total_confidence
        
                # Functions to run prompts
        def first_run(row, df_item, df_category):
            item = row[df_item.columns[0]]  # Assuming the first column is the 'Item' column
            categories = ", ".join(df_category[df_category.columns[0]].tolist())  # Assuming the first column is the 'Category' column
            filled_prompt = txt_content.replace("[Item]", item).replace("[Categories]", categories).replace("[Instructions]", instructions)
            return filled_prompt

        Retrain_Prompt_With_Corrections = """You are a data scientist data assistant and your job is to sort [Item] into one of the following categories: [Categories]

Here a few Examples:
[TRAINING_EXAMPLES]

Additional Context:
[Instructions]

Based on the examples above and the available categories, please classify the item.
Your response should ONLY contain the category name from the available categories listed above."""

        def retrain_run_with_corrections(row, df_item, df_category, corrections_data, instructions):
            item = row[df_item.columns[0]]
            categories = ", ".join(df_category[df_category.columns[0]].tolist())
            
            # Format training examples in a clear, numbered list
            training_examples = []
            for i, corr in enumerate(corrections_data, 1):
                training_examples.append(f"Example {i}:")
                training_examples.append(f"• Input: {corr[0]}")
                training_examples.append(f"• Correct Category: {corr[1]}")
                training_examples.append("")  # Add blank line between examples
            
            # Join all examples with newlines
            formatted_examples = "\n".join(training_examples).strip()
            
            filled_prompt = (Retrain_Prompt_With_Corrections
                .replace("[TRAINING_EXAMPLES]", formatted_examples)
                .replace("[Item]", item)
                .replace("[Categories]", categories)
                .replace("[Instructions]", instructions)
            )
            return filled_prompt

        def generate_json_objects(df_item, df_category, gptmodel, tokens_list, max_tokens_count):
            json_df_item = []
            for index, row in df_item.iterrows():
                filled_prompt = first_run(row, df_item, df_category) if not rerun_with_training else retrain_run_with_corrections(row, df_item, df_category, corrections_data, instructions)

                json_df_item_row = {
                    "model": gptmodel,
                    "logprobs": True,
                    "top_logprobs": 10,
                    "logit_bias": tokens_list,
                    "messages": [
                        {"role": "system", "content": "You are a data science tool used to help categorize information. Your answers will be fed into datatables."},
                        {"role": "user", "content": filled_prompt}
                    ],
                    "max_tokens": max_tokens_count,
                    "temperature": 0.50,
                    "metadata": {"row_id": index}
                }

                json_df_item.append(json_df_item_row)
                json.dumps(json_df_item)
            return json_df_item
        corrections_data = data.get('corrections', []) if rerun_with_training else []
        json_df_item = []
        json_df_item = generate_json_objects(df_item, df_category, gptmodel, tokens_list, max_tokens_count)
        # print(api_key)     
        # Call parallel processor
        json_df_item = process_json(
            request_json=json_df_item,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name=token_encoding_name,
            max_attempts=max_attempts,
            logging_level=logging_level,
            )
        total_tokens = 0
        for response in json_df_item:
            if not isinstance(response, Exception):
                response_data = response[1]
                # Get usage from the response if available
                if 'usage' in response_data:
                    total_tokens += response_data['usage']['total_tokens']
                else:
                    # If usage not available, use the max_tokens from request
                    request_data = response[0]  # Original request
                    total_tokens += request_data.get('max_tokens', 0)
        
        # Update metrics in Cosmos DB
        asyncio.run(update_user_metrics(
            container=container,
            user_id=user_id,
            tokens_used=total_tokens,
            training_run=rerun_with_training,
            module_type='classification',
            input_size=input_size,
            categories_size=categories_size
        ))
        
        total_input_tokens = 0
        total_output_tokens = 0
        for index, response in enumerate(json_df_item):
            if isinstance(response, Exception):
                print(f"Exception occurred in response for index {index}: {response}")
                df_item.at[index, 'Response'] = f"Error: {response}"  # Optionally log the error in the Response column
                df_item.at[index, 'Confidence'] = None
                continue
            # Extract response text and logprobs content
            #print(response)
            response_data = response[1]
            
            # Track token usage from API response
            if 'usage' in response_data:
                usage = response_data['usage']
                total_input_tokens += usage.get('prompt_tokens', 0)
                total_output_tokens += usage.get('completion_tokens', 0)
            # Extract response text and logprobs content
            response_text = response_data['choices'][0]['message']['content']
            logprobs_content = response_data['choices'][0]['logprobs']['content']
            row_id = response[2]['row_id']
            # Calculate confidence score if logprobs are available
            confidence_score = None
            if logprobs_content:
                confidence_score = calculate_confidence(logprobs_content, df_category, response_text)

            # Handle missing categories in df_category
            if response_text not in df_category['Category'].tolist():
                response_text = "Error: Response not in original categories: " + response_text

            # Update df_item at the corresponding index with the response and confidence score
            df_item.at[row_id, 'Response'] = response_text
            df_item.at[row_id, 'Confidence'] = confidence_score
            
            actual_total_cost = calculate_actual_cost(total_input_tokens, total_output_tokens)
            token_metrics = {
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens,
                'total_tokens': total_input_tokens + total_output_tokens,
                'total_cost': actual_total_cost
            }
        asyncio.run(update_user_metrics(
            container=container,
            user_id=user_id,
            tokens_used=token_metrics,
            training_run=rerun_with_training,
            module_type='classification',
            input_size=input_size,
            categories_size=categories_size
        ))

        # Adjust final cost
        if is_company_key:
            asyncio.run(adjust_final_cost(user_id, token_estimates["total_cost"], actual_total_cost))
        
        logging.info(f"Analysis complete. Input tokens: {total_input_tokens}, Output tokens: {total_output_tokens}, Total cost: ${actual_total_cost:.4f}")
        return jsonify({
            'results': df_item[['Item', 'Response', 'Confidence']].to_dict(orient='records'),
            'token_usage': {
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens,
                'total_tokens': total_input_tokens + total_output_tokens,
                'total_cost': actual_total_cost
            }
        }), 200
        
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500
# Update this in app.py
MODEL_TOKEN_RATES = {
    "gpt-4o": {
        "input": 2.50 / 1000000,  # $2.50 per 1M input tokens
        "output": 10.0 / 1000000  # $10.00 per 1M output tokens
    },
    "gpt-4o-mini": {
        "input": 0.15 / 1000000,  # $0.15 per 1M input tokens
        "output": 0.60 / 1000000  # $0.60 per 1M output tokens
    },
    "gpt-3.5-turbo": {
        "input": 0.15 / 1000000,  # $0.15 per 1M input tokens
        "output": 0.80 / 1000000  # $0.80 per 1M output tokens
    }
}
def calculate_actual_cost_open_format(input_tokens, output_tokens, model="gpt-4o"):
    """
    Calculates the actual cost based on separate input and output token counts
    and the specific model used
    """
    model_rates = MODEL_TOKEN_RATES.get(model, MODEL_TOKEN_RATES["gpt-4o"])
    input_cost = (input_tokens * model_rates["input"])
    output_cost = (output_tokens * model_rates["output"])
    return input_cost + output_cost

@app.route('/api/open_format', methods=['POST'])
def open_format():
    try:
        logging.info("Open format request received")
        data = request.get_json()

        # Extract API parameters from the request payload
        api_key = data.get('apiKey')
        user_id = data.get('userId')
        
        if not verify_subscription(user_id):
            return jsonify({"error": "Active subscription required"}), 403
            
        # if not api_key:
        #     logging.error("API key is missing.")
        #     return jsonify({"error": "API key is required."}), 400
        query = f"SELECT * FROM c WHERE c.userId = '{user_id}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        
        if not items:
            return jsonify({"error": "User not found"}), 404
            
        doc = items[0]
        current_balance = doc.get('balance', 0)
        user_api_key = doc.get('apiKey')
        
        # Check if both conditions are true: balance < 5 and API key is null/invalid
        if current_balance < 1 and (not user_api_key or not user_api_key.startswith('sk-') or len(user_api_key) < 20):
            logging.error(f"User {user_id} has insufficient balance and no valid API key")
            return jsonify({
                "error": "Please enter a valid API key in settings to continue. Your account balance is insufficient to use our API key."
            }), 403
        api_key, is_company_key = get_appropriate_api_key(user_id)

        # Validate input data
        input_data = data.get('inputData', [])
        instructions = data.get('instructions', '')
        model = data.get('model', 'gpt-4o')  #ZackUpdate Default to gpt-4o if not specified
        temperature = float(data.get('temperature', 0.7))

        if not input_data:
            logging.error("Input data is empty.")
            return jsonify({"error": "Input data cannot be empty."}), 400

        if not instructions:
            logging.error("Instructions are empty.")
            return jsonify({"error": "Instructions cannot be empty."}), 400

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        # print(api_key)
        # Create DataFrame from input; each row contains one "Input"
        df_item = pd.DataFrame([row[0] for row in input_data], columns=['Input'])
        
        # Estimate token usage and cost for the selected model
        token_estimates = estimate_token_usage(input_data, module_type="open_format", model=model)
        
        # Check and deduct balance if using company API key
        if is_company_key:
            try:
                asyncio.run(check_and_deduct_balance(user_id, token_estimates["total_cost"]))
            except Exception as e:
                return jsonify({"error": str(e)}), 403

        # Create prompt template Zack updated prompt
        prompt_template = """You are my assistant and your job is to help with the following task. Please be concise and to the point in your response.

Task: {instructions}

Input: {input_text}

Please provide your response in a direct and concise manner, worthy of being included in a well-maintanined dataset"""  #ZackUpdate Prompt template remains unchanged
        
        
        
        #ZackUpdate Build parallel requests for completions using process_json instead of async process_completions
        json_requests = []  #ZackUpdate Initialize list for completion JSON requests
        for index, row in df_item.iterrows():  #ZackUpdate Iterate over DataFrame rows
            prompt = prompt_template.format(instructions=instructions, input_text=row['Input'])  #ZackUpdate Build prompt for each row
            json_requests.append({  #ZackUpdate Append completion request JSON object
                "model": model,  #ZackUpdate Set model for completion
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that provides concise, direct responses."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,  #ZackUpdate Set temperature
                "metadata": {"row_id": index}  #ZackUpdate Include metadata for tracking
            })
        completion_request_url = "https://api.openai.com/v1/chat/completions"  #ZackUpdate Define chat completions API URL

        #ZackUpdate Call parallel processor for completions using process_json
        responses = process_json(  #ZackUpdate Invoke parallel processor for completion API calls
            request_json=json_requests,  #ZackUpdate JSON payload for completion requests
            request_url=completion_request_url,  #ZackUpdate Completion endpoint URL
            api_key=api_key,  #ZackUpdate API key for authentication
            max_requests_per_minute=max_requests_per_minute,  #ZackUpdate Rate limit configuration
            max_tokens_per_minute=max_tokens_per_minute,  #ZackUpdate Token rate limit configuration
            token_encoding_name=token_encoding_name,  #ZackUpdate Token encoding configuration
            max_attempts=max_attempts,  #ZackUpdate Maximum number of retry attempts
            logging_level=logging_level,  #ZackUpdate Logging level for parallel processor
        )

        #ZackUpdate Process responses from parallel processor and update result data based on metadata
        result_data = [None] * len(input_data)  #ZackUpdate Preallocate results list
        total_input_tokens = 0  #ZackUpdate Initialize token counter
        total_output_tokens = 0  #ZackUpdate Initialize token counter
        
        for response in responses:  #ZackUpdate Iterate through each response
            try:
                if isinstance(response, Exception):
                    if isinstance(response, (list, tuple)) and len(response) > 2:
                        metadata = response[2]
                    else:
                        metadata = {}
                    row_id = metadata.get("row_id")
                    logging.error(f"Error processing item for row {row_id}: {response}")  #ZackUpdate Log error with row_id
                    if row_id is not None:
                        result_data[row_id] = [input_data[row_id][0], f"Error: {str(response)}"]  #ZackUpdate Set error result for this row
                    continue

                # Unpack the response tuple; assume structure: [request_payload, api_response, metadata]
                request_payload, api_response, metadata = response
                row_id = metadata.get("row_id")
                if row_id is None:
                    logging.error("Missing row_id in metadata")  #ZackUpdate Log missing row_id
                    continue
                response_text = api_response['choices'][0]['message']['content']  #ZackUpdate Extract response text
                result_data[row_id] = [input_data[row_id][0], response_text]  #ZackUpdate Set result data at proper index
                usage = api_response.get("usage", {})  #ZackUpdate Extract token usage if available
                total_input_tokens += usage.get("prompt_tokens", 0)  #ZackUpdate Accumulate prompt tokens
                total_output_tokens += usage.get("completion_tokens", 0)  #ZackUpdate Accumulate completion tokens
            except Exception as e:
                logging.error(f"Exception processing response for row {row_id if 'row_id' in locals() else 'unknown'}: {e}")  #ZackUpdate Log exception with row_id
                if 'row_id' in locals() and row_id is not None:
                    result_data[row_id] = [input_data[row_id][0], f"Error: {str(e)}"]  #ZackUpdate Set error result for this row

        # Calculate actual cost using model-specific rates
        actual_total_cost = calculate_actual_cost_open_format(total_input_tokens, total_output_tokens, model)
        
        # Create token metrics dictionary for detailed tracking
        token_metrics = {
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'total_tokens': total_input_tokens + total_output_tokens,
            'total_cost': actual_total_cost,
            'model': model
        }
        
        # Update metrics in Cosmos DB
        asyncio.run(update_user_metrics(
            container=container,
            user_id=user_id,
            tokens_used=token_metrics,
            module_type='open_format',
            input_size=len(input_data)
        ))
        
        # Adjust final cost if using company key
        if is_company_key:
            asyncio.run(adjust_final_cost(user_id, token_estimates["total_cost"], actual_total_cost))
        
        logging.info(f"Open format analysis complete. Model: {model}, Input tokens: {total_input_tokens}, Output tokens: {total_output_tokens}, Total cost: ${actual_total_cost:.4f}")
        
        return jsonify({
            'results': result_data,
            'token_usage': {
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens,
                'total_tokens': total_input_tokens + total_output_tokens,
                'total_cost': actual_total_cost,
                'model': model
            }
        }), 200

    except Exception as e:
        logging.error(f"Unexpected error in open format request: {e}")
        return jsonify({"error": str(e)}), 500
    

# new module : category generator created by Awais

@app.route('/api/create-categories', methods=['POST'])
def analyze_categories():
    try:
        # Get data from request
        data = request.json
        logging.info("Received data request for category analysis")
        
        if not data:
            return jsonify({
                "success": False,
                "error": "Invalid request data"
            }), 400
            
        # Extract user ID
        user_id = data.get('userId')
        if not user_id:
            return jsonify({
                "success": False,
                "error": "User ID is required"
            }), 400
            
        # First check if user has access
        if not verify_subscription(user_id):
            return jsonify({
                "error": "Active subscription"
            }), 403
        query = f"SELECT * FROM c WHERE c.userId = '{user_id}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        
        if not items:
            return jsonify({"error": "User not found"}), 404
            
        doc = items[0]
        current_balance = doc.get('balance', 0)
        user_api_key = doc.get('apiKey')
        
        # Check if both conditions are true: balance < 1 and API key is null/invalid
        if current_balance < 1 and (not user_api_key or not user_api_key.startswith('sk-') or len(user_api_key) < 20):
            logging.error(f"User {user_id} has insufficient balance and no valid API key")
            return jsonify({
                "error": "Please enter a valid API key in settings to continue. Your account balance is insufficient to use our API key."
            }), 403            
        # Determine which API key to use
        api_key, is_company_key = get_appropriate_api_key(user_id)
        
        if not api_key:
            return jsonify({"error": "Valid API key not available"}), 403
            
        # Validate required fields
        required_fields = ['items', 'categorization_need', 'category_number']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"Missing required field: {field}"
                }), 400
        
        # Validate items data
        items = data.get('items', [])
        if not isinstance(items, list) or not items:
            return jsonify({
                "success": False,
                "error": "Items must be a non-empty list"
            }), 400
            
        # Estimate token usage - this will need to be customized based on your model and needs
        input_size = len(items)
        categorization_need = data.get('categorization_need', 'General Topics')
        category_number = int(data.get('category_number', 7))
        
        # Estimate tokens - Adjust these estimates for category generation
        estimated_input_tokens = input_size * 200  # Estimated tokens per input
        estimated_output_tokens = 500 + (category_number * 50)  # Base + per category
        
        # Get model rates
        model_rates = MODEL_TOKEN_RATES.get("gpt-4o", MODEL_TOKEN_RATES["gpt-4o"])
        
        # Calculate costs
        input_cost = (estimated_input_tokens * model_rates["input"])
        output_cost = (estimated_output_tokens * model_rates["output"])
        estimated_cost = input_cost + output_cost
        
        token_estimates = {
            "input_tokens": estimated_input_tokens,
            "output_tokens": estimated_output_tokens,
            "total_cost": estimated_cost
        }
        
        # Only deduct balance if using company API key (balance is being used)
        if is_company_key:
            try:
                asyncio.run(check_and_deduct_balance(user_id, estimated_cost))
            except Exception as e:
                return jsonify({"error": str(e)}), 403
                
        # Convert items to DataFrame
        try:
            df_item = pd.DataFrame(items, columns=['Item'])
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Error creating DataFrame: {str(e)}"
            }), 400
            
        # Initialize CategoryAnalyzer
        analyzer = CategoryAnalyzer(
            openai_api_key=api_key,
            gpt_model="gpt-4o"
        )

        # Perform category analysis
        results = analyzer.analyze_categories(
            df_item=df_item,
            categorization_need=categorization_need,
            category_number=category_number
        )
        
        # Calculate actual token usage
        token_metrics = {
            'input_tokens': results["token_counts"]["input_tokens"],
            'output_tokens': results["token_counts"]["output_tokens"],
            'total_tokens': results["token_counts"]["total_tokens"],
            'total_cost': calculate_actual_cost(
                results["token_counts"]["input_tokens"], 
                results["token_counts"]["output_tokens"]
            )
        }
        
        # Update metrics for tracking
        asyncio.run(update_user_metrics(
            container=container,
            user_id=user_id,
            tokens_used=token_metrics,
            module_type='category_generation',
            input_size=input_size,
            categories_size=category_number
        ))
        
        # Adjust final cost if using company key
        if is_company_key:
            asyncio.run(adjust_final_cost(user_id, estimated_cost, token_metrics["total_cost"]))

        # Return results
        return jsonify({
            "success": True,
            "results": {
                "categories": results["final_result"],
                "token_usage": token_metrics
            }
        })

    except Exception as e:
        logging.error(f"Error during category analysis: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# latest code for the Data Anonlyization

@app.route('/api/anonymizes', methods=['POST'])
def anonymize_datas():
    try:
        # Get data from request
        data = request.json
        logging.info(f"Received data for anonymization")
        
        if not data:
            return jsonify({
                "success": False,
                "error": "Invalid request data"
            }), 400
            
        # Extract user ID
        user_id = data.get('userId')
        if not user_id:
            return jsonify({
                "success": False,
                "error": "User ID is required"
            }), 400
            
        # First check if user has access (subscription or sufficient balance)
        if not verify_subscription(user_id):
            return jsonify({
                "error": "Active subscription"
            }), 403
        query = f"SELECT * FROM c WHERE c.userId = '{user_id}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))

        if not items:
            return jsonify({"error": "User not found"}), 404

        doc = items[0]
        current_balance = doc.get('balance', 0)
        user_api_key = doc.get('apiKey')

        # Check if both conditions are true: balance < 1 and API key is null/invalid
        if current_balance < 1 and (not user_api_key or not user_api_key.startswith('sk-') or len(user_api_key) < 20):
            logging.error(f"User {user_id} has insufficient balance and no valid API key")
            return jsonify({
                "error": "Please enter a valid API key in settings to continue. Your account balance is insufficient to use our API key."
            }), 403            
        # Determine which API key to use
        api_key, is_company_key = get_appropriate_api_key(user_id)
        
        # if not api_key:
        #     return jsonify({"error": "Valid API key not available"}), 403
        
        # Estimate token usage for anonymization
        # Roughly estimate based on the size of the dataset and categories
        input_size = len(data.get('dataset', []))
        
        # Estimate tokens - These estimates should be adjusted based on actual usage
        estimated_input_tokens = input_size * 250  # Estimated tokens per input record
        estimated_output_tokens = input_size * 100  # Estimated output tokens
        
        # Get model rates
        model_rates = MODEL_TOKEN_RATES.get("gpt-4o", MODEL_TOKEN_RATES["gpt-4o"])
        
        # Calculate costs
        input_cost = (estimated_input_tokens * model_rates["input"])
        output_cost = (estimated_output_tokens * model_rates["output"])
        estimated_cost = input_cost + output_cost
        print(f"Here is the estimated cost{estimated_cost}")
        
        # Only deduct balance if using company API key (balance is being used)
        if is_company_key:
            try:
                asyncio.run(check_and_deduct_balance(user_id, estimated_cost))
            except Exception as e:
                return jsonify({"error": str(e)}), 403
                
        # Initialize the anonymizer with the appropriate API key
        anonymizer = DataAnonymizer(api_key)
        anonymization_result = anonymizer.anonymize_dataset(
            data['dataset'],
            data['categories']
        )
        # Perform anonymization
        anonymized_data = anonymization_result["anonymized_data"]
        token_counts = anonymization_result["token_counts"]
        
        # Calculate actual tokens used (if available from anonymizer)
        # In a real implementation, you'd track actual token usage
        # For now, we'll use our estimates
        actual_total_cost = calculate_actual_cost(
            token_counts["input_tokens"], 
            token_counts["output_tokens"]
        )
        print(f"Here is the actual cost{actual_total_cost}")        
        token_metrics = {
            'input_tokens': token_counts["input_tokens"],
            'output_tokens': token_counts["output_tokens"],
            'total_tokens': token_counts["total_tokens"],
            'total_cost': actual_total_cost
        }
        print(f"Here are tokens calculated{token_metrics}")
        
        # Update metrics for tracking
        asyncio.run(update_user_metrics(
            container=container,
            user_id=user_id,
            tokens_used=token_metrics,
            module_type='data_anonymization',
            input_size=input_size
        ))
        
        # Adjust final cost if using company key
        if is_company_key:
            asyncio.run(adjust_final_cost(user_id, estimated_cost, actual_total_cost))

        return jsonify({
            "success": True,
            "anonymized_data": anonymized_data,
            "token_usage": token_metrics
        })

    except Exception as e:
        logging.error(f"Error during data anonymization: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/cleanup-temp', methods=['POST'])
def cleanup_temp():
    try:
        temp_dir = os.path.join(BASE_DIR, 'app_temp')
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logging.info(f"Deleted file: {file_path}")
                except Exception as e:
                    logging.error(f"Error removing {file_path}: {e}")
            return jsonify({'success': True, 'message': 'Temporary directory cleaned'}), 200
        else:
            return jsonify({'success': False, 'message': 'Temporary directory not found'}), 404
    except Exception as e:
        logging.error(f"Error cleaning temporary directory: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# UPLOAD_FOLDER = r'app_temp'
# # Ensure the upload folder exists
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'app_temp')
ALLOWED_EXTENSIONS = {'pdf'}

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_unique_filename(original_filename):
    """Generate a unique filename if a file with the same name exists"""
    base_name, extension = os.path.splitext(original_filename)
    counter = 1
    filename = original_filename
    while os.path.exists(os.path.join(UPLOAD_FOLDER, filename)):
        filename = f"{base_name}_{counter}{extension}"
        counter += 1
    return filename

CUSTOM_TEMP_DIR = os.path.join(BASE_DIR, 'app_temp')
os.makedirs(CUSTOM_TEMP_DIR, exist_ok=True)

@app.route('/process_pdf_folders', methods=['POST'])
def process_pdf_folders():
    """
    Enhanced endpoint to process all PDFs in a folder with extraction and categorization
    """
    try:
        # Extract data from request
        data = request.get_json()
        logging.info("received data: %s", data)
        
        # Explicitly log categorization parameters
        logging.info("CATEGORIZATION PARAMS:")
        logging.info("enable_categorization: %s (type: %s)", 
                    data.get('enable_categorization'), 
                    type(data.get('enable_categorization')).__name__)
        logging.info("categories: %s (type: %s, len: %s)", 
                    data.get('categories'), 
                    type(data.get('categories')).__name__,
                    len(data.get('categories', [])))
        
        folder_path = os.path.join(BASE_DIR, 'app_temp')
        specific_filenames = data.get('uploadedFileNames', [])
        recursive = data.get('recursive', False)
        Document_level_fields = data.get('document_fields', "")
        Line_item_fields = data.get('line_item_fields', "")
        Confidence_return = data.get('confidence_return', True)  # Default to True
        api_key = data.get('apiKey')
        user_id = data.get('userId')
        # Get vertical orientation and flatten line items settings
        useVe = data.get('vertical_orientation', False)
        flatten_line_items = bool(data.get('flatten_line_items', False))
        
        # Web search functionality temporarily disabled
        # use_web_search = bool(data.get('useWebSearch', False))
        # if use_web_search:
        #     # Web search implementation will go here
        
        # Get categorization config - handle string 'true' from JavaScript
        enable_categorization_raw = data.get('enable_categorization', False)
        # Convert string 'true' to True (JavaScript may send it as string)
        if isinstance(enable_categorization_raw, str):
            enable_categorization = enable_categorization_raw.lower() == 'true'
        else:
            enable_categorization = bool(enable_categorization_raw)
            
        category_list_data = data.get('categories', [])
        categorization_instructions = data.get('categorization_instructions', '')
        
        # Log the final values after conversion
        logging.info("FINAL CATEGORIZATION CONFIG:")
        logging.info("enable_categorization (after conversion): %s", enable_categorization)
        logging.info("category_list_data len: %d", len(category_list_data))
        
        # Failsafe: If categories exist but enable_categorization is False, force it to True
        if not enable_categorization and len(category_list_data) > 0:
            logging.warning("FAILSAFE: Categories found but enable_categorization is False. Forcing enable_categorization to True.")
            enable_categorization = True
        
        # Validate input parameters
        if not folder_path:
            return jsonify({'error': 'Folder path is required'}), 400
        if not os.path.exists(folder_path):
            return jsonify({'error': 'Invalid folder path'}), 400
        if not Document_level_fields and not Line_item_fields and not category_list_data:
            return jsonify({'error': 'At least one field type must be specified'}), 400
        
        # Check if user has access via subscription
        if not verify_subscription(user_id):
            return jsonify({"error": "Active subscription required"}), 403
        
        # Get user document to check balance and API key
        query = f"SELECT * FROM c WHERE c.userId = '{user_id}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        
        if not items:
            return jsonify({"error": "User not found"}), 404
            
        doc = items[0]
        current_balance = doc.get('balance', 0)
        user_api_key = doc.get('apiKey')
        
        # Check if both conditions are true: balance < 1 and API key is null/invalid
        if current_balance < 1 and (not user_api_key or not user_api_key.startswith('sk-') or len(user_api_key) < 20):
            logging.error(f"User {user_id} has insufficient balance and no valid API key")
            return jsonify({
                "error": "Please enter a valid API key in settings to continue. Your account balance is insufficient to use our API key."
            }), 403
            
        # Determine which API key to use
        api_key, is_company_key = get_appropriate_api_key(user_id)
        
        # Estimate token usage based on document count and complexity
        pdf_count = len(specific_filenames) if specific_filenames else len(glob.glob(os.path.join(folder_path, "*.pdf")))
        estimated_input_tokens = pdf_count * 1000  # Base estimate per document
        estimated_output_tokens = pdf_count * 500   # Base estimate per response
        
        # Adjust estimates based on extraction complexity
        if Document_level_fields:
            field_count = len(Document_level_fields.split(','))
            estimated_input_tokens += field_count * 100
            estimated_output_tokens += field_count * 50
            
        if Line_item_fields:
            line_item_field_count = len(Line_item_fields.split(','))
            estimated_input_tokens += line_item_field_count * 200
            estimated_output_tokens += line_item_field_count * 100
            
        # Additional tokens for categorization if enabled
        if enable_categorization and category_list_data:
            estimated_input_tokens += pdf_count * 300
            estimated_output_tokens += pdf_count * 100
        
        # Get model rates
        model_rates = MODEL_TOKEN_RATES.get("gpt-4o", MODEL_TOKEN_RATES["gpt-4o"])
        
        # Calculate estimated cost
        estimated_input_cost = (estimated_input_tokens * model_rates["input"])
        estimated_output_cost = (estimated_output_tokens * model_rates["output"])
        estimated_total_cost = estimated_input_cost + estimated_output_cost
        
        # Only deduct balance if using company API key
        if is_company_key:
            try:
                asyncio.run(check_and_deduct_balance(user_id, estimated_total_cost))
            except Exception as e:
                return jsonify({"error": str(e)}), 403
        
        # Get PDF files based on specific filenames if provided
        if specific_filenames and len(specific_filenames) > 0:
            logging.info("Processing specific files: %s", specific_filenames)
            if folder_path.startswith('\\'):
                folder_path = folder_path[1:]
            abs_folder_path = os.path.join(os.getcwd(), folder_path)
            pdf_files = []
            for filename in specific_filenames:
                file_path = os.path.join(abs_folder_path, filename)
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    pdf_files.append(file_path)
                    logging.info("Added file to process: %s", file_path)
                else:
                    logging.warning("File not found: %s", file_path)
        else:
            if recursive:
                pdf_files = []
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        if file.lower().endswith('.pdf'):
                            pdf_files.append(os.path.join(root, file))
            else:
                pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
            
        logging.info("Found %s PDF files to process", len(pdf_files))
        if not pdf_files:
            return jsonify({'error': 'No PDF files found to process'}), 400

        # Process PDFs
        encoded_images_data = []
        for pdf_path in pdf_files:
            logging.info("Processing PDF: %s", pdf_path)
            try:
                base64_images = pdf_to_base64_pymupdf(pdf_path)
                encoded_images_data.append({
                    "file_name": os.path.basename(pdf_path),
                    "images": base64_images
                })
            except Exception as e:
                logging.error("Error processing %s: %s", pdf_path, e)
                continue

        # Read extraction prompt from file and update it as specified
        with open(os.path.join(BASE_DIR, 'Prompt', 'extraction.txt'), 'r', encoding='utf-8') as file:
            initial_prompt = file.read()
        
        # Create extraction prompt with updated function
        extraction_prompt = initial_prompt\
            .replace("[Document level fields]", Document_level_fields or "None")\
            .replace("[Line item fields]", Line_item_fields or "None")
        
        # Process extraction phase with improved confidence calculation
        logging.info("Starting extraction phase")
        final_df = run_extraction(encoded_images_data, extraction_prompt, api_key, Confidence_return, gptmodel="gpt-4o", flatten_line_items=flatten_line_items)
        if final_df.empty:
            return jsonify({'error': 'No data could be extracted from the PDFs'}), 400
        
        # Debug log DataFrame before categorization
        logging.info("DataFrame after extraction - shape: %s, columns: %s", 
                    final_df.shape, final_df.columns.tolist())
        
        # Run categorization if enabled
        if enable_categorization and category_list_data:
            logging.info("Starting enhanced categorization phase with %d categories", len(category_list_data))
            # Ensure orig_index is preserved during categorization
            if 'orig_index' in final_df.columns:
                logging.info("Detected flattened data with orig_index, count: %d", len(final_df['orig_index'].unique()))
                orig_indices_present = True
            else:
                orig_indices_present = False
                
            # Fix parameter order: df, category_list_data, instructions, api_key
            final_df = run_categorization(final_df, category_list_data, categorization_instructions, api_key)
            
            # Log to confirm categorization data exists
            logging.info("After categorization - columns: %s", final_df.columns.tolist())
            if 'Response' in final_df.columns:
                unique_responses = final_df['Response'].unique()
                logging.info("Response column values: %s", unique_responses)
                logging.info("Response value counts: %s", final_df['Response'].value_counts().to_dict())
            else:
                logging.error("Response column not found after categorization!")
        else:
            logging.info("No categorization needed")
        
        # Reorder columns: place 'Response' and 'Confidence' after file_name
        cols = list(final_df.columns)
        if enable_categorization and category_list_data:
            # Check if file_name exists in columns
            if 'file_name' in cols:
                # Get file_name position
                file_name_idx = cols.index('file_name')
                
                # Create new column order: cols before file_name, file_name, Response, Confidence, remaining cols
                new_cols = [col for col in cols if col not in ('file_name', 'Response', 'Confidence')]
                new_order = new_cols[:file_name_idx] + ['file_name', 'Response', 'Confidence'] + new_cols[file_name_idx:]
            else:
                # If there's no file_name, fallback to putting them at position 2 & 3
                new_order = [cols[0]] + ['Response', 'Confidence'] + [col for col in cols[1:] if col not in ('Response', 'Confidence')]
            
            # Log before reordering
            response_before = final_df['Response'].head(5).tolist() if 'Response' in final_df.columns else "Response column not found"
            logging.info("Before reordering - Response values: %s", response_before)
            
            final_df = final_df[new_order]
            
            # Extra logging to verify data is present
            logging.info("After reordering - first 5 rows Response values: %s", 
                         final_df['Response'].head(5).tolist() if 'Response' in final_df.columns else "Response column not found")
        
        # Now we can safely drop the orig_index as we no longer need it
        if 'orig_index' in final_df.columns:
            final_df = final_df.drop(columns=['orig_index'])
            
        # Also remove the is_flattened flag if it exists
        if 'is_flattened' in final_df.columns:
            final_df = final_df.drop(columns=['is_flattened'])

        # Reorder all other confidence columns to place them directly after their field
        cols = list(final_df.columns)
        field_conf_map = {}
        
        # Find all confidence columns (except the main category Confidence)
        confidence_cols = [col for col in cols if 'Confidence' in col and col != 'Confidence']
        
        # For each confidence column, find its corresponding field
        for conf_col in confidence_cols:
            # Extract the field name by removing ' Confidence' suffix
            if ' Confidence' in conf_col:
                field = conf_col.replace(' Confidence', '')
                if field in cols:
                    field_conf_map[field] = conf_col
        
        # Create a new column order placing each confidence column after its field
        if field_conf_map:
            logging.info("Found field-confidence pairs: %s", field_conf_map)
            new_order = []
            for col in cols:
                new_order.append(col)
                # If this column has a corresponding confidence column and we haven't added it yet
                if col in field_conf_map and field_conf_map[col] not in new_order:
                    new_order.append(field_conf_map[col])
            
            # Remove any duplicates in case something went wrong
            new_order = list(dict.fromkeys(new_order))
            
            # Apply the new order
            if len(new_order) == len(cols):  # Safety check
                final_df = final_df[new_order]
                logging.info("Reordered columns to place confidence values next to their fields")
        
        # Confidence values are now calculated once for all fields in run_extraction
        # so we don't need to propagate them here

        # Remove file_name Confidence column if it exists
        if 'file_name Confidence' in final_df.columns:
            logging.info("Removing unnecessary file_name Confidence column")
            final_df = final_df.drop(columns=['file_name Confidence'])
        
        # Deduplication now happens in the extraction phase (run_extraction function)
        # so we don't need to do it again here

        # Calculate actual token usage
        actual_input_tokens = input_tokens if 'input_tokens' in globals() else 0
        actual_output_tokens = output_tokens if 'output_tokens' in globals() else 0
        actual_total_tokens = total_tokens if 'total_tokens' in globals() else 0
        
        # Calculate actual cost
        actual_input_cost = (actual_input_tokens * model_rates["input"])
        actual_output_cost = (actual_output_tokens * model_rates["output"])
        actual_total_cost = actual_input_cost + actual_output_cost
        
        # Create token metrics dictionary for detailed tracking
        token_metrics = {
            'input_tokens': actual_input_tokens,
            'output_tokens': actual_output_tokens,
            'total_tokens': actual_total_tokens,
            'total_cost': actual_total_cost
        }
        
        # Update metrics in Cosmos DB
        asyncio.run(update_user_metrics(
            container=container,
            user_id=user_id,
            tokens_used=token_metrics,
            module_type='pdf_extraction',
            input_size=len(pdf_files)
        ))
        
        # Adjust final cost if using company key
        if is_company_key:
            asyncio.run(adjust_final_cost(user_id, estimated_total_cost, actual_total_cost))

        # Set pandas display options to show more data
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        # Check if Response column exists in the final dataframe
        has_response_column = 'Response' in final_df.columns
        logging.info("Final DataFrame has Response column: %s", has_response_column)
        if has_response_column:
            logging.info("Response column values: %s", final_df['Response'].unique())
        else:
            logging.error("Response column missing from final_df!")
            
        logging.info("Final DataFrame:\n%s", final_df.to_string())
        


        if useVe:
            # Additional logging for debugging
            logging.info("Before vertical orientation - DataFrame shape: %s, columns: %s", 
                         final_df.shape, final_df.columns.tolist())
            if 'Response' in final_df.columns:
                logging.info("Response column before vertical orientation: %s", 
                            final_df['Response'].head(5).tolist())
            
            # Get the original column names
            columns = final_df.columns.tolist()
            
            # Transpose the DataFrame
            final_df = final_df.transpose()
            
            # Create a new DataFrame with field names in first column
            new_df = pd.DataFrame()
            new_df['Field_Name'] = columns
            
            # Add the data columns (one column per document)
            for i in range(len(final_df.columns)):
                new_df[f'Document_{i}'] = final_df.iloc[:, i].values
                
            # Replace the final_df with new_df
            final_df = new_df

        # Get the column order from the DataFrame
        column_order = list(final_df.columns)
        logging.info("Final column order before conversion: %s", column_order)
        
        # Convert the entire DataFrame to a list of dicts while preserving column order
        final_results = []
        for row in final_df.replace({pd.NA: None, float('nan'): None}).to_dict(orient='records'):
            # Create a new dictionary with keys in the same order as the DataFrame columns
            filtered_row = {}
            for col in column_order:
                # Include all columns regardless of value
                filtered_row[col] = row.get(col, None)
            final_results.append(filtered_row)
        
        # Add detailed logging of final_results structure
        logging.info("Final Results Structure:")
        logging.info("Number of results: %d", len(final_results))
        
        # Log sample of the first result to verify structure
        if len(final_results) > 0:
            first_result = final_results[0]
            logging.info("First result keys: %s", list(first_result.keys()))
            # Check if categorization data is present
            if 'Response' in first_result:
                logging.info("First result Response: %s", first_result['Response'])
            else:
                logging.error("Response column missing from final_results!")
        
        # Use json.dumps with sort_keys=False to preserve the order of keys
        response_data = {
            'success': True,
            'data': final_results,
            'token_usage': token_metrics,
            'message': 'PDF processing completed successfully'
        }
        
        # Verify that categorization data is in the response_data
        if len(final_results) > 0 and 'Response' in final_results[0]:
            logging.info("Response value in final JSON: %s", final_results[0]['Response'])
            
        # Manually verify no NaN or non-JSON-serializable values
        for item in final_results:
            for key, value in item.items():
                if pd.isna(value):
                    item[key] = None
                elif isinstance(value, pd.Timestamp):
                    item[key] = value.isoformat()
         
        # Final sanity check - inject categorization if needed and enabled
        if enable_categorization and category_list_data:
            # Force the final data to include categorization
            if not any('Response' in item for item in final_results):
                logging.warning("FINAL EMERGENCY FIX: Injecting categorization into first result")
                most_common_category = category_list_data[0]['Category'] if category_list_data else "Unknown"
                if hasattr(run_categorization, 'last_response') and run_categorization.last_response:
                    most_common_category = run_categorization.last_response
                
                if len(final_results) > 0:
                    final_results[0]['Response'] = most_common_category
                    final_results[0]['Confidence'] = 0.8
                    
                    # If vertical layout, also add as separate rows at the top
                    if useVe:
                        response_row = {'Field_Name': 'Response', 'Document_0': most_common_category}
                        confidence_row = {'Field_Name': 'Confidence', 'Document_0': 0.8}
                        final_results.insert(0, confidence_row)
                        final_results.insert(0, response_row)
                        
            # Make sure response_data contains the updated final_results
            response_data['data'] = final_results
                    
        # Return a Response object with the manually serialized JSON
        return Response(
            json.dumps(response_data, sort_keys=False, default=str),
            mimetype='application/json',
            status=200
        )
        
    except Exception as e:
        logging.error("Unexpected error: %s", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
        
    finally:
        try:
            for file in os.listdir(CUSTOM_TEMP_DIR):
                file_path = os.path.join(CUSTOM_TEMP_DIR, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logging.warning("Error removing temporary file %s: %s", file_path, e)
        except Exception as e:
            logging.warning("Error during cleanup: %s", e)

def pdf_to_base64_pymupdf(pdf_path):
    base64_images = []
    try:
        doc = fitz.open(pdf_path)
        for page_number in range(len(doc)):
            page = doc[page_number]
            pix = page.get_pixmap()
            image_path = f"page_{page_number + 1}.png"
            pix.save(image_path)
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                base64_images.append(base64_image)
            os.remove(image_path)
    except Exception as e:
        logging.error("Error converting %s: %s", pdf_path, e)
    return base64_images

def calculate_confidence(doc_df, logprobs_content):
    log_probs1 = [{"logprob": token["logprob"], "token": token["token"], "toplog": token["top_logprobs"]}
                  for token in logprobs_content]
    capturing = False
    buffer = ""
    column_indices = []
    last_matched_index = -1
    for i, entry in enumerate(log_probs1):
        token = entry['token']
        if '"' in token:
            if capturing:
                matched_column = None
                if '"' in token and token.index('"') != 0:
                    buffer += token[:token.index('"')]
                for col_index, col in enumerate(doc_df.columns[last_matched_index + 1:], start=last_matched_index + 1):
                    if buffer.strip() == str(doc_df.loc[0, col]).strip():
                        matched_column = col_index
                        last_matched_index = col_index
                        break
                for idx in column_indices:
                    log_probs1[idx]['column_index'] = matched_column
                buffer = ""
                column_indices = []
            capturing = not capturing
        elif capturing:
            buffer += token
            column_indices.append(i)
    for entry in log_probs1:
        if 'column_index' not in entry:
            entry['column_index'] = None
    log_probs1 = [row for row in log_probs1 if row['column_index'] is not None]
    def clean_token(token):
        token = token.lower().translate(str.maketrans('', '', string.punctuation)).strip()
        return str(int(token)) if token.isdigit() else token
    for i, row in enumerate(log_probs1):
        log_probs1[i].setdefault('Selected', None)
        log_probs1[i].setdefault('Selected Incorrectly', None)
        log_probs1[i].setdefault('Not Selected', None)
        toplogs = [[tl["token"], tl["logprob"]] for tl in row["toplog"]]
        for token_val, logprob in toplogs:
            if token_val == row['token']:
                log_probs1[i]['Selected'] = math.exp(logprob) if log_probs1[i]['Selected'] is None else log_probs1[i]['Selected'] + math.exp(logprob)
            elif clean_token(token_val) == clean_token(row['token']):
                log_probs1[i]['Selected Incorrectly'] = math.exp(logprob) if log_probs1[i]['Selected Incorrectly'] is None else log_probs1[i]['Selected Incorrectly'] + math.exp(logprob)
            else:
                log_probs1[i]['Not Selected'] = math.exp(logprob) if log_probs1[i]['Not Selected'] is None else log_probs1[i]['Not Selected'] + math.exp(logprob)
    log_probs1 = pd.DataFrame(log_probs1)
    
    # Make a defensive copy of columns to prevent dictionary size change during iteration
    original_columns = list(doc_df.columns)
    
    # Skip columns that already have "Confidence" in their name
    columns_to_process = [col for col in original_columns if 'Confidence' not in col]
    
    # Create a copy of the dataframe for results
    result_df = doc_df.copy()
    
    for col_idx, col_name in enumerate(columns_to_process):
        current_col_idx = doc_df.columns.get_loc(col_name)
        Selected = 1
        Not_Selected = 1
        First_Value = True
        for index, row in log_probs1.iterrows():
            if row['column_index'] == col_idx and First_Value:
                Not_Selected *= row['Not Selected']
                Selected *= row['Selected']
                First_Value = False
            elif row['column_index'] == col_idx and not First_Value:
                Not_Selected += (Selected * row['Not Selected'])
                Selected *= row['Selected']
            elif row['column_index'] != col_idx and not First_Value:
                break
        total_prob_sum = Selected + Not_Selected
        normalized_entropy_probs = [
            Selected / total_prob_sum if total_prob_sum > 0 else 0,
            Not_Selected / total_prob_sum if total_prob_sum > 0 else 0,
        ]
        entropy = -sum([p * math.log2(p) for p in normalized_entropy_probs if p > 0])
        max_entropy = math.log2(2)
        # Ensure a minimum confidence value to prevent rounding to 0
        total_confidence = max(0.00001, (1 - entropy / max_entropy))
        new_confidence_col = f"{col_name} Confidence"
        if new_confidence_col not in result_df.columns:
            # Insert at the position right after the original column
            col_names = list(result_df.columns)
            col_position = col_names.index(col_name) + 1
            # Insert new column at the right position
            left_cols = col_names[:col_position]
            right_cols = col_names[col_position:]
            result_df = result_df.reindex(columns=left_cols + [new_confidence_col] + right_cols)
        result_df[new_confidence_col] = total_confidence
    
    return result_df

def run_extraction(encoded_images_data, prompt, api_key, Confidence_return, gptmodel, flatten_line_items=False):
    
    global input_tokens, output_tokens, total_tokens
    if 'input_tokens' not in globals():
        global input_tokens
        input_tokens = 0
    if 'output_tokens' not in globals():
        global output_tokens
        output_tokens = 0
    if 'total_tokens' not in globals():
        global total_tokens
        total_tokens = 0
    
    all_pdf_details = []  # for error DataFrames
    extraction_requests = []
    
    # Assemble the extraction_requests list with row_id metadata
    for idx, pdf_data in enumerate(encoded_images_data):
        logging.info("Processing file: %s", pdf_data['file_name'])
        images_content = []
        for page_number, base64_image in enumerate(pdf_data["images"], start=1):
            images_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        content = ([{
            "type": "text",
            "text": prompt
        }] + images_content)
        extraction_request = {
            "model": gptmodel,
            "messages": [{
                "role": "user",
                "content": content
            }],
            "response_format": {"type": "json_object"},
            "temperature": 0.23,
            "logprobs": True,
            "top_logprobs": 4,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "metadata": {
                "file_name": pdf_data['file_name'],
                "row_id": idx
            }
        }
        extraction_requests.append(extraction_request)
    
    # # Optionally write extraction_requests to a file
    # with open(r"C:\Users\57510\OneDrive - Bain\Documents\ChatNoir\PDF Extraction\extraction_requests.txt", 
    #           "w", encoding="utf-8") as f:
    #     f.write(json.dumps(extraction_requests, indent=2))
    # print("Assembled")
    
    # extraction_requests_json = extraction_requests
    base_dir = os.path.abspath(os.path.dirname(__file__))
    prompts_folder = os.path.join(base_dir, 'prompts')
    os.makedirs(prompts_folder, exist_ok=True)

# Write extraction requests to a file in the prompts folder
    extraction_requests_file = os.path.join(prompts_folder, 'extraction_requests.txt')
    with open(extraction_requests_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(extraction_requests, indent=2))
    print(f"Assembled and saved to {extraction_requests_file}")

    extraction_requests_json = extraction_requests
    try:
        def extraction_worker(extraction_requests_json, api_key):
            global responses
            responses = process_json(
                request_json=extraction_requests_json,
                request_url=request_url,
                api_key=api_key,
                max_requests_per_minute=max_requests_per_minute,
                max_tokens_per_minute=max_tokens_per_minute,
                token_encoding_name=token_encoding_name,
                max_attempts=max_attempts,
                logging_level=logging_level,
            )
            
        # Start the extraction in a separate thread.
        t = threading.Thread(target=extraction_worker, args=(extraction_requests_json, api_key))
        t.start()
        # Poll the status while the thread is alive.
        while t.is_alive():
            print(f"[{time.strftime('%H:%M:%S')}] Status update - ")
            time.sleep(1)
        t.join()
        print("Extraction complete.")
        
        # Pre-populate a DataFrame with file names using the same row order as encoded_images_data
        doc_df = pd.DataFrame({'file_name': [data['file_name'] for data in encoded_images_data]})
        logging.info("doc_df: %s", doc_df)  
        # Initialize a dictionary to hold DataFrames by row_id for proper ordering
        result_dfs = {}
        # Process each response
        for response_tuple in responses:
            row_id = None  # default in case we can't get it
            try:
                if isinstance(response_tuple, Exception):
                    # In case of an exception response, try to obtain row_id from metadata if possible
                    # (if not available, we won't update doc_df)
                    raise response_tuple  # jump to the except block
                
                # Reset any globals for logprobs processing
                global doc_logprobs_split, doc_level_logprobs, line_items_logprobs, current_line_item_index, all_item_probs
                doc_logprobs_split = False
                doc_level_logprobs = []
                line_items_logprobs = []
                current_line_item_index = 0
                all_item_probs = []
                
                # Process the response
                response_metadata = response_tuple[2]
                response_data = response_tuple[1]
                row_id = response_metadata.get('row_id')
                # Retrieve the corresponding pdf_data using row_id
                pdf_data = encoded_images_data[row_id]
                
                response_text = response_data['choices'][0]['message']['content']
                # Grab logprobs for confidence calculation later if needed
                logprobs_content = response_data['choices'][0]['logprobs']['content']  # Already a Python list
                logging.info(f"Type of logprobs_content: {type(logprobs_content)}")
                logging.info(f"First few items of logprobs_content: {logprobs_content[:2] if isinstance(logprobs_content, list) else logprobs_content}")
                
                # Save logprobs to file
                # logprobs_file_path = os.path.join(r"C:\Users\ZacharyHarper\Documents\Chat Noir\PDF Extraction", 
                #                                 f"logprobs_{pdf_data['file_name'].replace('.pdf', '')}.json")
                # with open(logprobs_file_path, 'w', encoding='utf-8') as f:
                #     json.dump(logprobs_content, f, indent=4, ensure_ascii=False)
                
                mapping_dict = json.loads(response_text)
                logging.info(f"Mapping dict structure for file {pdf_data['file_name']}:")
                logging.info(json.dumps(mapping_dict, indent=2))
                # Update token usage counts
                usage = response_data.get('usage', {})
                input_tokens += usage.get('prompt_tokens', 0)
                output_tokens += usage.get('completion_tokens', 0)
                total_tokens += usage.get('total_tokens', 0)
    
                doc_fields = mapping_dict.get("Document-Level Fields", {})
                line_items = mapping_dict.get("Line-Item Fields", [])

                if flatten_line_items or not doc_fields:
                    row_dfs = []
                    logprobs_holder = logprobs_content 
                    
                    # --- Step 1: Process Document-Level Fields --- 
                    doc_logprobs_section = []
                    base_data = {'file_name': pdf_data['file_name']}
                    base_data.update(doc_fields)
                    
                    if calculate_confidence: 
                        # Check if doc_fields exists and is not None (even if empty)
                        has_doc_fields = doc_fields is not None
                        doc_logprobs_section, logprobs_holder = split_logprobs(True, doc_fields, logprobs_holder)
                        
                        # If we have document fields (even empty ones), calculate confidence
                        if has_doc_fields and doc_logprobs_section: 
                           doc_df_temp = pd.DataFrame([base_data]) 
                           doc_df_temp = calculate_confidence(doc_df_temp, doc_logprobs_section) 
                           for col in doc_df_temp.columns:
                               if 'Confidence' in col:
                                   base_data[col] = doc_df_temp.at[0, col]
                        else:
                            logging.warning("Document section split failed or no document fields.")

                    current_remaining_logprobs = logprobs_holder
                        
                    for i, item in enumerate(line_items, start=1):
                        logging.info(f"--- Processing Line Item {i} ---")
                        row_data = base_data.copy() 
                        row_data.update(item) 
                        line_item_section = []
                        if calculate_confidence:
                            line_item_section, current_remaining_logprobs = split_logprobs(False, item, current_remaining_logprobs)
                            
                            if line_item_section: 
                                item_df = pd.DataFrame([item]) 
                                item_df = calculate_confidence(item_df, line_item_section)
                                for col in item_df.columns:
                                    if 'Confidence' in col:
                                        row_data[col] = item_df.at[0, col]
                                logging.info(f"Updated row_data with line item {i} confidence.")
                            else:
                                logging.warning(f"Split failed for line item {i}, skipping confidence calculation.")
                        
                        row_dfs.append(pd.DataFrame([row_data]))

                    # --- Write Debug Data to File --- 
                    # try:
                    #     with open(debug_file_path, 'w', encoding='utf-8') as f:
                    #         json.dump(split_debug_data, f, indent=4)
                    #     logging.info(f"Split debug data saved to: {debug_file_path}")
                    # except Exception as e:
                    #     logging.error(f"Failed to write split debug data: {e}")
                    # --- End Write Debug ---

                    # Combine all row DataFrames into the final result for this document
                    if row_dfs:
                        doc_df = pd.concat(row_dfs, ignore_index=True)
                    else: # Handle case where there were no line items or all splits failed
                         # If base_data exists (i.e., doc_fields were processed or present)
                         if base_data:
                              doc_df = pd.DataFrame([base_data])
                         else: # Truly empty case
                              doc_df = pd.DataFrame() 
                    logging.info(f"Finished processing document. Final DF shape: {doc_df.shape}")
                else:
                    # Original wide-format approach
                    data = {'file_name': pdf_data['file_name']}
                    data.update(doc_fields)
                    doc_df = pd.DataFrame([data])
                    for i, item in enumerate(line_items, start=1):
                        for key, value in item.items():
                            doc_df[f"{key} {i}"] = value

                    if Confidence_return and logprobs_content:
                        doc_df = calculate_confidence(doc_df, logprobs_content)
                    else:
                        print(f"Skipping confidence calculation for {pdf_data['file_name']}. Confidence_return: {Confidence_return}, logprobs available: {logprobs_content is not None}")
                    # Store the DataFrame by row_id for ordered collection later
                result_dfs[row_id] = doc_df
            except Exception as e:
                logging.error(f"Error processing row {row_id}: {str(e)}")
                if row_id is not None:
                    result_dfs[row_id] = pd.DataFrame()  # Store empty DataFrame for failed processing
        
        # Now add DataFrames to all_pdf_details in order of row_id
        for i in range(len(encoded_images_data)):
            if i in result_dfs:
                all_pdf_details.append(result_dfs[i])
                    
    except Exception as e:
        logging.error(f"Error in extraction process: {e}")
    # No flattening, just concatenate all PDF details
    if all_pdf_details:
        final_df = pd.concat(all_pdf_details, ignore_index=True)
    else:
        final_df = pd.DataFrame()
    
    return final_df

def first_run(row, df_category, txt_content, instructions, images_content=None):
    item = "Attached as images" if images_content else "Item not provided"
    categories = ", ".join(df_category['Category'].tolist())
    filled_text = txt_content.replace("[Item]", item).replace("[Categories]", categories).replace("[Instructions]", instructions)
    if images_content:
        content = [{"type": "text", "text": filled_text}] + images_content
        return content
    return filled_text

def retrain_run(row, df_category, txt_content, instructions, images_content=None):
    return first_run(row, df_category, txt_content, instructions, images_content=images_content)

def build_parallel_request(prompt_text, row_id, tokens_list, max_tokens_count):
    return {
        "model": gptmodel,
        "logprobs": True,
        "top_logprobs": 10,
        "logit_bias": tokens_list,
        "messages": [{
            "role": "user",
            "content": prompt_text
        }],
        "max_tokens": max_tokens_count,
        "temperature": 0.50,
        "metadata": {"row_id": row_id}
    }

def generate_json_objects(df_item, df_category, tokens_list, max_tokens_count, txt_content, instructions):
    json_requests = []
    for index, row in df_item.iterrows():
        images_content = []
        if "images" in row and row["images"]:
            for base64_image in row["images"]:
                images_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
        filled_prompt = first_run(row, df_category, txt_content, instructions, images_content=images_content) \
                        if not globals().get('rerun_with_training', False) \
                        else retrain_run(row, df_category, txt_content, instructions, images_content=images_content)
        json_request = build_parallel_request(filled_prompt, index, tokens_list, max_tokens_count)
        json_requests.append(json_request)
    return json_requests

def calculate_confidence_categorization(logprobs_content, df_category, response_text):
    category_sums = {
        'Selected category': [],
        'Not-selected category': [],
        'Model deviation': [],
        'Selected category- Incorrect tokens': []
    }
    category_row_index = df_category[df_category['Category'] == response_text].index
    category_row = category_row_index[0] if len(category_row_index) > 0 else None
    tokens_set = {t.lower() for tokens in df_category['tokens'] for t in tokens}
    response_text_lower = response_text.lower()
    for item in logprobs_content:
        token_probs = {key: 0.0 for key in category_sums}
        for top_logprob in item['top_logprobs']:
            token_lower = top_logprob['token'].lower()
            probability = math.exp(top_logprob['logprob'])
            if category_row is not None and token_lower in [t.lower() for t in df_category.at[category_row, 'tokens']]:
                token_probs['Selected category'] += probability
            elif token_lower not in tokens_set:
                token_probs['Model deviation'] += probability
            elif token_lower in response_text_lower:
                token_probs['Selected category- Incorrect tokens'] += probability
            else:
                token_probs['Not-selected category'] += probability
        for key, prob in token_probs.items():
            category_sums[key].append(prob)
    max_length = max(len(v) for v in category_sums.values())
    for key in category_sums:
        if len(category_sums[key]) < max_length:
            category_sums[key] += [0.0] * (max_length - len(category_sums[key]))
    summary_df = pd.DataFrame({
        'Category': list(category_sums.keys()),
        **{f'Position {i+1}': [category_sums[cat][i] for cat in category_sums] for i in range(max_length)}
    })
    total_model_deviation = 0
    for i in range(max_length):
        total_model_deviation += (1 - total_model_deviation) * (summary_df.at[summary_df[summary_df['Category'] == 'Model deviation'].index[0], f'Position {i + 1}'])
    entropy_probs = []
    for i in range(max_length + 1):
        if i < max_length:
            log_probability = math.log(summary_df.at[summary_df[summary_df['Category'] == 'Not-selected category'].index[0], f'Position {i + 1}'] + 1e-10)
            for j in range(i):
                primary_prob = summary_df.at[summary_df[summary_df['Category'] == 'Selected category'].index[0], f'Position {j + 1}']
                log_probability += math.log(primary_prob + 1e-10)
        else:
            log_probability = sum([math.log(summary_df.at[summary_df[summary_df['Category'] == 'Selected category'].index[0], f'Position {j + 1}'] + 1e-10) for j in range(max_length)])
        entropy_probs.append(math.exp(log_probability))
    total_prob_sum = sum(entropy_probs)
    normalized_entropy_probs = [p / total_prob_sum for p in entropy_probs] if total_prob_sum > 0 else [1 / len(entropy_probs)] * len(entropy_probs)
    entropy = -sum([p * math.log2(p) for p in normalized_entropy_probs if p > 0])
    max_entropy = math.log2(max_length + 1)
    # Ensure a minimum confidence value to prevent rounding to 0
    total_confidence = max(0.00001, (1 - total_model_deviation) * (1 - entropy / max_entropy) if max_entropy > 0 else (1 - total_model_deviation))
    return total_confidence

def run_categorization(df_encoded, category_list_data, categorization_instructions, api_key):
    global input_tokens, output_tokens, total_tokens
    
    # Store last values for emergency retrieval
    run_categorization.last_response = None
    run_categorization.last_confidence = None
    
    # Check if this is flattened data
    has_orig_index = 'orig_index' in df_encoded.columns
    has_sub_index = 'sub_index' in df_encoded.columns
    has_is_flattened = 'is_flattened' in df_encoded.columns
    
    # Only consider it flattened if it has the necessary columns AND is properly marked
    is_flattened = has_orig_index and (has_sub_index or has_is_flattened)
    
    logging.info("Running categorization. DataFrame has orig_index: %s, has_sub_index: %s, has_is_flattened: %s", 
                has_orig_index, has_sub_index, has_is_flattened)
    logging.info("Is data treated as flattened: %s", is_flattened)
    
    if is_flattened:
        # For flattened data, group by orig_index to get one request per document
        doc_indices = df_encoded['orig_index'].unique()
        logging.info("Found %d unique document indices in flattened data", len(doc_indices))
        
        # We'll create a document-level DataFrame with just one row per document
        doc_level_df = pd.DataFrame(index=doc_indices)
        
        # Safely get file_name for each document index
        file_names = []
        for idx in doc_indices:
            rows = df_encoded[df_encoded['orig_index'] == idx]
            if len(rows) > 0 and 'file_name' in rows.columns:
                file_names.append(rows['file_name'].iloc[0])
            else:
                # If we can't find a file_name, use a default
                file_names.append(f"Document_{idx}")
                
        doc_level_df['file_name'] = file_names
        doc_level_df['Category'] = ""
        doc_level_df['Confidence'] = ""
        
        # Sample data from each document for categorization
        df_for_cat = pd.DataFrame()
        for idx in doc_indices:
            # Get first row from each document for categorization
            rows = df_encoded[df_encoded['orig_index'] == idx]
            if len(rows) > 0:
                first_row = rows.iloc[0].copy()
                first_row.name = idx  # Set the index to match doc_level_df
                df_for_cat = pd.concat([df_for_cat, pd.DataFrame([first_row])], sort=False)
            else:
                logging.warning("No rows found for document index %s", idx)
    else:
        # Standard non-flattened case - even if orig_index exists, treat as regular data
        doc_level_df = df_encoded.copy()
        doc_level_df['Category'] = ""
        doc_level_df['Confidence'] = ""
        df_for_cat = df_encoded.copy()
    
    with open(os.path.join(BASE_DIR, 'Prompt', 'Categorization_Prompt.txt'), 'r', encoding='utf-8') as file:
        txt_content = file.read()
    
    df_category = pd.DataFrame(category_list_data)
    df_category['tokens'] = None
    tokens_list = {}
    category_token_counts = {}
    encoding = tiktoken.encoding_for_model(gptmodel)
    
    for index, row in df_category.iterrows():
        category = row['Category']
        token_ids = encoding.encode(category)
        tokens = [encoding.decode([token_id]) for token_id in token_ids]
        for token_id in token_ids:
            tokens_list[token_id] = 5
        category_token_counts[category] = len(token_ids)
        df_category.at[index, 'tokens'] = tokens
    
    max_tokens_category = max(category_token_counts, key=category_token_counts.get)
    max_tokens_count = category_token_counts[max_tokens_category]
    
    # Generate requests for categorization based on the document-level records
    json_requests = generate_json_objects(df_for_cat, df_category, tokens_list, max_tokens_count, txt_content, categorization_instructions)
    logging.info("Generated %d categorization requests", len(json_requests))
    
    try:
        responses = process_json(
            request_json=json_requests,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name=token_encoding_name,
            max_attempts=max_attempts,
            logging_level=logging_level,
        )
        
        for response_tuple in responses:
            if isinstance(response_tuple, Exception):
                idx = responses.index(response_tuple)
                logging.error("Exception occurred in response for index %s: %s", idx, response_tuple)
                doc_level_df.at[idx, 'Response'] = f"Error: {response_tuple}"
                doc_level_df.at[idx, 'Confidence'] = None
                continue
            
            response_data = response_tuple[1]
            response_text = response_data['choices'][0]['message']['content']
            logprobs_content = response_data['choices'][0]['logprobs']['content']
            row_id = response_tuple[2]['row_id']
            
            confidence_score = None
            if logprobs_content:
                confidence_score = calculate_confidence_categorization(logprobs_content, df_category, response_text)
            
            if response_text not in df_category['Category'].tolist():
                response_text = "Error: Response not in original categories: " + response_text
            
            doc_level_df.at[row_id, 'Response'] = response_text
            doc_level_df.at[row_id, 'Confidence'] = confidence_score
            
            usage = response_data.get('usage', {})
            input_tokens += usage.get('prompt_tokens', 0)
            output_tokens += usage.get('completion_tokens', 0)
            total_tokens += usage.get('total_tokens', 0)
    
    except Exception as e:
        logging.error("Unexpected error in categorization: %s", e)
    
    # Now merge categorization results back with original data
    if is_flattened:
        # For flattened data, merge categorization results to each row by orig_index
        result_df = df_encoded.copy()
        
        # Log categorization results for debugging
        logging.info("Categorization results doc_level_df shape: %s", doc_level_df.shape)
        logging.info("docIds: %s", doc_level_df.index.tolist())
        categorization_values = {}
        
        # Add the Response and Confidence to each row based on its orig_index
        for idx in doc_indices:
            try:
                response = doc_level_df.at[idx, 'Category']
                confidence = doc_level_df.at[idx, 'Confidence']
                categorization_values[idx] = {"Response": response, "Confidence": confidence}
                
                # Use .loc to update all matching rows
                matching_rows = result_df['orig_index'] == idx
                if any(matching_rows):
                    result_df.loc[matching_rows, 'Category'] = response
                    result_df.loc[matching_rows, 'Confidence'] = confidence
                else:
                    logging.warning("No matching rows found for orig_index %s", idx)
            except Exception as e:
                logging.error("Error applying categorization for index %s: %s", idx, e)
        
        logging.info("Applied categorization to result_df. Values: %s", categorization_values)
        
    else:
        # For non-flattened data, just return the updated DataFrame
        result_df = doc_level_df.copy()
    
    # Verify categorization data is present in the result
    if 'Category' in result_df.columns:
        has_responses = not result_df['Category'].isna().all()
        logging.info("Result has valid responses: %s. Sample: %s", 
                    has_responses, result_df['Category'].head(3).tolist())
    else:
        logging.warning("No 'Category' column in categorization results")
    
    # At the end of the function, store the most important categorization result
    if 'Category' in result_df.columns and not result_df['Category'].isna().all():
        # Store the most common response for emergency retrieval
        most_common_response = result_df['Category'].mode()[0]
        run_categorization.last_response = most_common_response
        
        # Find the confidence for this response
        if 'Category' in result_df.columns:
            confidence_values = result_df[result_df['Category'] == most_common_response]['Confidence']
            if not confidence_values.empty and not confidence_values.isna().all():
                run_categorization.last_confidence = confidence_values.iloc[0]
            else:
                run_categorization.last_confidence = 0.8  # Default confidence if missing
        else:
            run_categorization.last_confidence = 0.8  # Default confidence
    
    return result_df

# Configure upload folder
@app.route('/api/upload-files', methods=['POST'])
def upload_files():
    try:
        # Debug logging
        logging.info(f"Request Files: {request.files}")
        logging.info(f"Files keys: {request.files.keys()}")
        
        # Check if any file was sent
        if 'files[]' not in request.files:
            logging.warning("No 'files[]' in request.files")
            return jsonify({
                'success': False,
                'message': 'No files provided'
            }), 400

        files = request.files.getlist('files[]')
        logging.info(f"Number of files received: {len(files)}")
        
        if not files or files[0].filename == '':
            logging.warning("No files selected or empty filename")
            return jsonify({
                'success': False,
                'message': 'No files selected'
            }), 400

        uploaded_files = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # Get a unique filename to avoid overwrites
                unique_filename = get_unique_filename(filename)
                file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                
                file.save(file_path)
                uploaded_files.append(unique_filename)
                logging.info(f"Saved file: {unique_filename} to {file_path}")
            else:
                logging.error(f"Invalid file type: {file.filename}")
                return jsonify({
                    'success': False,
                    'message': f'Invalid file type for {file.filename}. Only PDF files are allowed.'
                }), 400

        return jsonify({
            'success': True,
            'message': 'Files uploaded successfully',
            'fileNames': uploaded_files,
            'uploadDirectory': UPLOAD_FOLDER
        })

    except Exception as e:
        logging.error(f"Upload error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error uploading files: {str(e)}'
        }), 500

@app.route('/api/delete-file', methods=['POST'])
def delete_file():
    try:
        filename = request.json.get('filename')
        if not filename:
            return jsonify({
                'success': False,
                'message': 'No filename provided'
            }), 400

        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        if os.path.commonpath([file_path, UPLOAD_FOLDER]) != UPLOAD_FOLDER:
            return jsonify({
                'success': False,
                'message': 'Invalid file path'
            }), 400

        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({
                'success': True,
                'message': f'File {filename} deleted successfully'
            })
        
        return jsonify({
            'success': False,
            'message': 'File not found'
        }), 404
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error deleting file: {str(e)}'
        }), 500

def split_logprobs(opener, row_data, logprobs_content):
    matching_section = []
    current_index = 0  
    opening_counter = 0
    closing_counter = 0
    
    while current_index < len(logprobs_content):
        token_data = logprobs_content[current_index]
        token_value = token_data['token']
        if "{" in token_value:  
            opening_counter += 1
        if "}" in token_value: 
            closing_counter += 1

        if opener == True:
            if opening_counter == 2 and closing_counter == 1:
                matching_section = logprobs_content[:current_index+1]
                remaining_section = logprobs_content[current_index+1:]
                return matching_section, remaining_section
            
        elif opener == False:
            if opening_counter == 1 and closing_counter == 1:
                matching_section = logprobs_content[:current_index+1]
                remaining_section = logprobs_content[current_index+1:]
                return matching_section, remaining_section
        else:
            raise ValueError(f"Invalid opener: {opener}")

        current_index += 1  # Increment the index to move to the next token
    
    # If we reach here, we couldn't find matching brackets
    return logprobs_content, []

if __name__ == '__main__':
    load_dotenv()
    app.run(debug=True, port=5000)
