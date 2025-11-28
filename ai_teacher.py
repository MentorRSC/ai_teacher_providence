import torch
import os
import json
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import speech_recognition as sr
import pyttsx3

class Phi3Teacher:
    def __init__(self, config_path: str = "config.json"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.vector_store = None
        self.speech_engine = None
        self.speech_recognizer = sr.Recognizer()
        
        print("🤖 Initializing AI Teacher with Phi-3-mini...")
        self._setup_models()
        self._setup_knowledge_base()
        self._setup_speech()
        print("✅ AI Teacher Ready!")
    
    def _setup_models(self):
        """Load Phi-3-mini model"""
        print("📥 Loading Phi-3-mini model...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model_name"],
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_name"],
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            print(f"✅ Phi-3-mini loaded on device: {self.model.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _setup_knowledge_base(self):
        """Process syllabus materials into searchable database"""
        print("📚 Setting up knowledge base...")
        
        if os.path.exists(self.config["vector_store_path"]):
            # Load existing vector store
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.vector_store = FAISS.load_local(
                self.config["vector_store_path"], 
                embeddings
            )
            print("✅ Loaded existing syllabus index")
        else:
            # Create new vector store
            self.vector_store = self._create_vector_store()
            print("✅ Created new syllabus index")
    
    def _create_vector_store(self) -> FAISS:
        """Process all syllabus documents"""
        documents = []
        syllabus_path = self.config["syllabus_path"]
        
        if not os.path.exists(syllabus_path):
            os.makedirs(syllabus_path)
            print(f"📁 Created syllabus directory: {syllabus_path}")
            print("⚠️  Please add your syllabus files (PDF/TXT) to this folder")
            return None
        
        # Load PDF files
        for filename in os.listdir(syllabus_path):
            if filename.endswith('.pdf'):
                filepath = os.path.join(syllabus_path, filename)
                try:
                    loader = PyPDFLoader(filepath)
                    documents.extend(loader.load())
                    print(f"✅ Loaded PDF: {filename}")
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
            
            # Load text files
            elif filename.endswith('.txt'):
                filepath = os.path.join(syllabus_path, filename)
                try:
                    loader = TextLoader(filepath)
                    documents.extend(loader.load())
                    print(f"✅ Loaded TXT: {filename}")
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
        
        if not documents:
            print("⚠️  No syllabus documents found!")
            return None
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        
        print(f"📖 Processed {len(chunks)} knowledge chunks")
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(self.config["vector_store_path"])
        
        return vector_store
    
    def _setup_speech(self):
        """Initialize text-to-speech engine"""
        try:
            self.speech_engine = pyttsx3.init()
            self.speech_engine.setProperty('rate', 150)  # Speaking speed
            self.speech_engine.setProperty('volume', 0.8)  # Volume level
        except Exception as e:
            print(f"⚠️  Speech engine initialization failed: {e}")
            self.speech_engine = None
    
    def _get_relevant_context(self, query: str, num_chunks: int = 3) -> str:
        """Retrieve relevant syllabus content for the query"""
        if self.vector_store is None:
            return "No syllabus content available."
        
        try:
            docs = self.vector_store.similarity_search(query, k=num_chunks)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context
        except Exception as e:
            return f"Error retrieving context: {e}"
    
    def _create_teaching_prompt(self, question: str, context: str) -> str:
        """Create a structured prompt for educational responses"""
        return f"""You are an expert teacher. Answer the student's question using ONLY the provided syllabus content.

SYLLABUS CONTENT:
{context}

STUDENT'S QUESTION: {question}

IMPORTANT INSTRUCTIONS:
- Answer strictly based on the syllabus content above
- If the answer isn't in the syllabus, say "This topic isn't covered in our syllabus materials"
- Structure your answer clearly
- Use simple, educational language
- Be encouraging and patient

TEACHER'S ANSWER:"""
    
    def ask_question(self, question: str) -> str:
        """Ask a question and get educational response"""
        if self.model is None:
            return "AI model not loaded. Please check initialization."
        
        print(f"🎓 Student: {question}")
        
        # Get relevant syllabus content
        context = self._get_relevant_context(question)
        
        # Create teaching prompt
        prompt = self._create_teaching_prompt(question, context)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.config["max_context_length"]
        ).to(self.model.device)
        
        # Generate response
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config["max_new_tokens"],
                    temperature=self.config["temperature"],
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the teacher's answer
            if "TEACHER'S ANSWER:" in full_response:
                response = full_response.split("TEACHER'S ANSWER:")[-1].strip()
            else:
                response = full_response
            
            return response
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def speak_response(self, text: str):
        """Convert text to speech"""
        print(f"🤖 Teacher: {text}")
        
        if self.speech_engine:
            try:
                self.speech_engine.say(text)
                self.speech_engine.runAndWait()
            except Exception as e:
                print(f"⚠️  Speech synthesis failed: {e}")
        else:
            print("🔇 Speech not available")
    
    def listen_to_question(self) -> str:
        """Listen to voice input and convert to text"""
        print("🎤 Listening... (Speak now)")
        
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                self.speech_recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Listen for audio
                audio = self.speech_recognizer.listen(source, timeout=10)
                
                # Convert to text
                text = self.speech_recognizer.recognize_google(audio)
                print(f"🗣️  You said: {text}")
                return text
                
        except sr.WaitTimeoutError:
            return "No speech detected. Please try again."
        except sr.UnknownValueError:
            return "Could not understand audio. Please speak clearly."
        except Exception as e:
            return f"Voice recognition error: {e}"

def main():
    """Main interaction loop"""
    print("=" * 60)
    print("           AI TEACHER WITH PHI-3-MINI")
    print("=" * 60)
    
    # Initialize teacher
    teacher = Phi3Teacher()
    
    # Display system info
    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"\nSystem Info:")
    print(f"  • Device: {device}")
    print(f"  • Model: Phi-3-mini (3.8B parameters)")
    print(f"  • Syllabus materials: {len(os.listdir(teacher.config['syllabus_path'])) if os.path.exists(teacher.config['syllabus_path']) else 0} files")
    
    print("\n🎯 Ready to teach! Choose input method:")
    print("1. Text input (type your question)")
    print("2. Voice input (speak your question)") 
    print("3. Exit")
    
    while True:
        try:
            choice = input("\n📝 Choose option (1-3): ").strip()
            
            if choice == "1":
                question = input("\n💬 Your question: ")
                if question.lower() in ['exit', 'quit']:
                    break
                
                response = teacher.ask_question(question)
                teacher.speak_response(response)
                
            elif choice == "2":
                question = teacher.listen_to_question()
                if question and not question.startswith("Error") and not question.startswith("No speech"):
                    response = teacher.ask_question(question)
                    teacher.speak_response(response)
                else:
                    print(f"❌ {question}")
                    
            elif choice == "3":
                print("\n👋 Thank you for learning! Goodbye!")
                break
                
            else:
                print("❌ Please choose 1, 2, or 3")
                
        except KeyboardInterrupt:
            print("\n\n🛑 Session interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
