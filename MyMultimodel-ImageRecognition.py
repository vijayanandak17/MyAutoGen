import os
import nest_asyncio
import asyncio
import mimetypes
from pathlib import Path
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Apply nest_asyncio at the top for Jupyter/IPython compatibility
nest_asyncio.apply()

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file.")


def get_mime_type(file_path: str) -> str:
    """Detect MIME type from file extension."""
    mime_type, _ = mimetypes.guess_type(file_path)
    
    # Fallback mapping for common image types
    if mime_type is None:
        ext = Path(file_path).suffix.lower()
        mime_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp'
        }
        mime_type = mime_map.get(ext, 'image/png')
    
    return mime_type


async def analyze_image(image_path: str):
    """Analyze an image using OpenAI multimodal model."""
    try:
        # Convert to absolute path and normalize
        image_path = os.path.abspath(os.path.expanduser(image_path))
        
        if not os.path.exists(image_path):
            print(f"⚠️ File not found: {image_path}")
            return
        
        # Validate it's a file (not a directory)
        if not os.path.isfile(image_path):
            print(f"⚠️ Path is not a file: {image_path}")
            return
        
        # Get correct MIME type
        mime_type = get_mime_type(image_path)
        print(f"📋 Detected MIME type: {mime_type}")
        
        # Read the image bytes
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        print(f"✅ Loaded image: {len(image_bytes)} bytes")
        
        # Initialize the model client
        model_client = OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key=api_key
        )
        
        # Create the assistant
        assistant = AssistantAgent(
            name="ImageAnalyzer",
            model_client=model_client,
            description="An AI assistant that describes and analyzes images."
        )
        
        print("🔍 Analyzing image...")
        
        # Convert image to base64 for embedding in message
        import base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Create message with embedded image - correct format for MultiModalMessage
        from autogen_agentchat.messages import MultiModalMessage
        from autogen_core import Image
        
        # Create Image object from bytes
        image_obj = Image.from_base64(image_base64)
        
        # MultiModalMessage with text and Image object
        task_message = MultiModalMessage(
            content=[
                "Describe this image in detail. Identify key objects, scene, and any text visible in it.",
                image_obj
            ],
            source="user"
        )
        
        result = await assistant.run(
            task=task_message
        )
        
        print("\n🖼️ Assistant's analysis:")
        print(result.messages[-1].content)
        print("-" * 60)
        
    except FileNotFoundError:
        print(f"❌ File not found: {image_path}")
    except PermissionError:
        print(f"❌ Permission denied reading file: {image_path}")
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main function to handle user input and run analysis."""
    print("=" * 60)
    print("🖼️  Image Analyzer with AutoGen & GPT-4o-mini")
    print("=" * 60)
    
    image_path = input("\n📸 Enter image file path: ").strip().strip('"').strip("'")
    
    if not image_path:
        print("⚠️ No path provided.")
        return
    
    await analyze_image(image_path)


if __name__ == "__main__":
    # Simple, clean event loop handling
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️ Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()