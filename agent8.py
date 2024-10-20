import os
import logging
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai, deepgram, silero, elevenlabs

# Set up logging
logger = logging.getLogger("voice-agent")

# Load environment variables
load_dotenv(dotenv_path=".env.local")

def get_pdf_text(pdf_directory):
    text = ""
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

global_conversation_chain = None

def create_vectorstore_and_chain(text_chunks):
    global global_conversation_chain
    logger.info("Creating vector store...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    
    logger.info("Creating conversation chain...")
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    global_conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return global_conversation_chain

# Process PDF and create conversation chain at startup
logger.info("Processing PDF...")
raw_text = get_pdf_text("pdfs")
text_chunks = get_text_chunks(raw_text)
conversation_chain = create_vectorstore_and_chain(text_chunks)
logger.info("PDF processing and conversation chain creation completed.")

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


class PDFAwareLLM(openai.LLM):
    async def complete(self, prompt: str, **kwargs) -> str:
        global global_conversation_chain
        # Use the conversation chain to get a response based on the PDF content
        response = global_conversation_chain({"question": prompt})
        
        # The response from the vector store is in the 'answer' key
        vector_response = response['answer']
        
        # You can add additional processing here if needed
        final_response = vector_response
        
        return final_response

async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a customer representative working for Bank of Cyprus. Your interface with users will be voice. "
            "You should use short and concise responses, and avoid usage of unpronounceable punctuation. "
            "You have access to information from a PDF document. When asked questions, use this information to provide answers. Do not assume things."
            "If you don't find relevant information, say you don't have an answer right now and will forward the inquiry to another representative and ask if there is a need to help with something else related to banking services."
            "If you do't find relevant information, do not tell the user to do any action."
            "Do not mention anything about the document. Say only that you don't know if you don't know the answer."
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    conversation_active = True

    async def on_speech_recognized(text: str):
        nonlocal conversation_active
        try:
            if text.lower() in ["goodbye", "bye", "exit", "quit"]:
                await assistant.say("Thank you for calling us. Goodbye!")
                conversation_active = False
                return

            # Process user input and generate response
            response = await assistant.llm.complete(text)

            # Speak the response
            await assistant.say(response, allow_interruptions=False)
        except Exception as e:
            logger.error(f"Error processing speech: {e}")
            await assistant.say("I'm sorry, I encountered an error. Could you please repeat your question?")
    
    assistant = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=PDFAwareLLM(model="gpt-4-turbo"),
        tts=elevenlabs.TTS(model_id="eleven_turbo_v2"),
        chat_ctx=initial_ctx,
    )

    assistant.start(ctx.room, participant)

    # Greet the user
    await assistant.say("Hello! Thank you for calling Bank of Cyprus. How can I help you today?", allow_interruptions=False)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )