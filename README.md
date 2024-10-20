# FinZy

Description: AI agent that acts as customer service representative. It accepts phone calls from the customer and answers questions based on information from a pdf that serves 
as his knowledge base. 

Stack: Python, LiveKit, Telnyx, OpenAI, ElevenLabs, Deepgram, Langchain

To start with LiveKit, check its documentation: https://docs.livekit.io/home/

SIP Trunk provider: Telnyx, but can also work perfectly with Twilio. Tested with both. A number must be purchased from these providers. To setup your number
on LiveKit, check the Telephony part of LiveKit's docs. 

A folder named "pdfs" must be created in the same directory.

All imported libraries must be installed using the "pip install <library_name>" command 

Create the a venv and run inside to avoid any conflicts with the packages

The API keys used, are stored separately on a .env.local file in the same directory
