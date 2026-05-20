import os
import torch
from PIL import Image
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, CallbackContext
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv("C:/Users/ajays/OneDrive/Desktop/.env")

# Get API keys from environment variables
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TOKEN or not GEMINI_API_KEY:
    raise ValueError("❌ Missing API keys! Set TELEGRAM_BOT_TOKEN and GEMINI_API_KEY in the .env file.")

# Load BLIP model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
user_chatting = {}

# Gemini text generation helper
def get_gemini_response(prompt):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text if hasattr(response, 'text') and response.text else "I'm here to help!"

# Gemini-powered sarcasm/mixed detection
def analyze_sarcasm_with_gemini(text):
    prompt = f"""Analyze the sentiment of the following text. Tell me if it is SARCASM, MIXED, POSITIVE, or NEGATIVE. Be honest if it's hard to tell.

Text: "{text}"

Respond with only one word: SARCASM, MIXED, POSITIVE, or NEGATIVE."""
    response = get_gemini_response(prompt)
    return response.strip().upper()

# Main sentiment analyzer
def get_sentiment(msg):
    result = sentiment_analyzer(msg)[0]
    label = result['label'].upper()

    # If label is neutral or confidence is low, ask Gemini
    if label == "NEUTRAL" or result['score'] < 0.75:
        label = analyze_sarcasm_with_gemini(msg)

    return label

# Analyze image -> caption -> sentiment
def analyze_image_sentiment(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    sentiment = get_sentiment(caption)
    return caption, sentiment

# Telegram utilities
async def send_message(update, text, reply_markup=None):
    max_length = 4096
    for i in range(0, len(text), max_length):
        if isinstance(update, Update):
            await update.message.reply_text(text[i:i + max_length], reply_markup=reply_markup if i == 0 else None)
        else:
            await update.message.reply_text(text[i:i + max_length], reply_markup=reply_markup if i == 0 else None)

def get_main_menu():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🧠 Analyze Sentiment", callback_data='analyze')],
        [InlineKeyboardButton("🖼️ Analyze Image", callback_data='analyze_image')],
        [InlineKeyboardButton("ℹ️ Help", callback_data='help')]
    ])

def get_back_menu():
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Menu", callback_data='menu')]])

def get_negative_response_menu():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💪 Get Motivation", callback_data='motivation')],
        [InlineKeyboardButton("😂 Tell a Joke", callback_data='joke')],
        [InlineKeyboardButton("💬 Chat with Bot", callback_data='chat')],
        [InlineKeyboardButton("🔙 Back to Menu", callback_data='menu')]
    ])

# Handlers
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("👋 Welcome! Please choose an option:", reply_markup=get_main_menu())

async def button(update: Update, context: CallbackContext):
    query = update.callback_query
    await query.answer()

    if query.data == 'chat':
        user_chatting[query.from_user.id] = True
        await send_message(query, "You can now chat with me! Type 'exit' to stop.", reply_markup=get_back_menu())
    elif query.data == 'motivation':
        response = get_gemini_response("Give me a motivational message.")
        await send_message(query, response, reply_markup=get_back_menu())
    elif query.data == 'joke':
        response = get_gemini_response("Tell me a funny joke.")
        await send_message(query, response, reply_markup=get_back_menu())
    elif query.data == 'help':
        await send_message(query, "ℹ️ Use the menu to chat, get motivated, laugh, or analyze sentiment. Send a photo or message to begin!", reply_markup=get_back_menu())
    elif query.data == 'menu':
        await send_message(query, "👋 Welcome back! Please choose an option:", reply_markup=get_main_menu())
    elif query.data == 'analyze':
        await send_message(query, "🧠 Send me a message and I’ll analyze its sentiment.", reply_markup=get_back_menu())
    elif query.data == 'analyze_image':
        await send_message(query, "📷 Send me an image and I’ll analyze its sentiment.", reply_markup=get_back_menu())
    else:
        await send_message(query, "Invalid option!", reply_markup=get_main_menu())

async def handle_message(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    user_message = update.message.text
    if user_id in user_chatting and user_message.lower() != "exit":
        response = get_gemini_response(user_message)
        await send_message(update, response)
        return
    if user_message.lower() == "exit":
        user_chatting.pop(user_id, None)
        await send_message(update, "Chat session ended.", reply_markup=get_main_menu())
        return
    sentiment = get_sentiment(user_message)
    keyboard = get_negative_response_menu() if sentiment in ["NEGATIVE", "MIXED", "SARCASM"] else get_back_menu()
    await send_message(update, f"Detected Sentiment: {sentiment}", reply_markup=keyboard)

async def handle_photo(update: Update, context: CallbackContext):
    photo = update.message.photo[-1]
    photo_file = await photo.get_file()
    image_path = f"temp_{update.message.message_id}.jpg"
    await photo_file.download_to_drive(image_path)

    try:
        caption, sentiment = analyze_image_sentiment(image_path)
        keyboard = get_negative_response_menu() if sentiment in ["NEGATIVE", "MIXED", "SARCASM"] else get_back_menu()
        message = f"🖼️ Image Caption: {caption}\n🧠 Detected Sentiment: {sentiment}"
        await send_message(update, message, reply_markup=keyboard)
    except Exception as e:
        print(f"Error analyzing image: {e}")
        await send_message(update, "Sorry, I couldn't process the image.", reply_markup=get_main_menu())
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

async def error_handler(update: object, context: CallbackContext):
    print(f"⚠️ Error: {context.error}")
    if isinstance(update, Update) and update.message:
        await send_message(update, "An error occurred. Please try again later.")

# Main function
def main():
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CallbackQueryHandler(button))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)
    print("🚀 Bot is running...")
    application.run_polling()

if __name__ == '__main__':
    main()
