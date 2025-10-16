import os
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file.")


async def simple_agent_demo():
    """Simple demo of running and observing AutoGen agents."""
    
    print("=" * 70)
    print("🤖 AutoGen Agent Demo - Running and Observing Agents")
    print("=" * 70)
    print()
    
    # Initialize the model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=api_key
    )
    
    # Create a Research Agent
    research_agent = AssistantAgent(
        name="ResearchAgent",
        model_client=model_client,
        description="An AI assistant that researches topics and provides information.",
        system_message="You are a helpful research assistant. Provide clear, concise information."
    )
    
    # Create a Writer Agent
    writer_agent = AssistantAgent(
        name="WriterAgent",
        model_client=model_client,
        description="An AI assistant that writes content based on research.",
        system_message="You are a creative writer. Take information and write it in an engaging way."
    )
    
    # Create a team with round-robin chat
    team = RoundRobinGroupChat(
        participants=[research_agent, writer_agent],
        max_turns=4  # Limit conversation turns
    )
    
    # Task for the agents
    task = """
    Research the benefits of Python programming language and then 
    write a short paragraph about why beginners should learn Python.
    """
    
    print(f"📋 Task: {task.strip()}")
    print()
    print("🔄 Agents starting conversation...")
    print("-" * 70)
    print()
    
    # Run the team and observe the conversation
    result = await Console(team.run_stream(task=task))
    
    print()
    print("-" * 70)
    print("✅ Conversation completed!")
    print()
    
    # Display final result
    print("📝 Final Result:")
    print("=" * 70)
    if hasattr(result, 'messages') and result.messages:
        final_message = result.messages[-1]
        print(f"From: {final_message.source}")
        print(f"Message: {final_message.content}")
    
    return result


async def single_agent_demo():
    """Simple single agent demo with observation."""
    
    print("=" * 70)
    print("🤖 Single Agent Demo - Observing Agent Responses")
    print("=" * 70)
    print()
    
    # Initialize the model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=api_key
    )
    
    # Create an assistant agent
    assistant = AssistantAgent(
        name="Assistant",
        model_client=model_client,
        description="A helpful AI assistant.",
        system_message="You are a helpful assistant. Provide clear, concise answers."
    )
    
    # Ask a question
    question = "Explain what AutoGen is in 2-3 sentences."
    
    print(f"❓ Question: {question}")
    print()
    print("🤔 Agent is thinking...")
    print("-" * 70)
    print()
    
    # Run and observe
    result = await assistant.run(task=question)
    
    print("💬 Agent Response:")
    print("=" * 70)
    print(result.messages[-1].content)
    print()
    
    # Show metadata
    print("📊 Response Metadata:")
    print(f"   - Total messages: {len(result.messages)}")
    print(f"   - Stop reason: {result.stop_reason}")
    print()
    
    return result


async def interactive_demo():
    """Interactive demo where you can chat with an agent."""
    
    print("=" * 70)
    print("💬 Interactive Agent Chat - Type 'exit' to quit")
    print("=" * 70)
    print()
    
    # Initialize the model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=api_key
    )
    
    # Create an assistant agent
    assistant = AssistantAgent(
        name="ChatAssistant",
        model_client=model_client,
        description="A conversational AI assistant.",
        system_message="You are a friendly, helpful assistant. Keep responses concise."
    )
    
    conversation_history = []
    
    while True:
        # Get user input
        user_input = input("👤 You: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\n👋 Goodbye! Thanks for chatting.")
            break
        
        if not user_input:
            continue
        
        print("🤖 Agent: ", end="", flush=True)
        
        # Run the agent
        result = await assistant.run(task=user_input)
        response = result.messages[-1].content
        
        print(response)
        print()
        
        conversation_history.append({
            "user": user_input,
            "agent": response
        })
    
    # Summary
    if conversation_history:
        print(f"\n📊 Session Summary: {len(conversation_history)} exchanges")


async def main():
    """Main function to run demos."""
    
    print("\n🎯 Select a demo to run:")
    print("1. Single Agent Demo")
    print("2. Multi-Agent Team Demo")
    print("3. Interactive Chat")
    print()
    
    choice = input("Enter choice (1-3): ").strip()
    print()
    
    try:
        if choice == "1":
            await single_agent_demo()
        elif choice == "2":
            await simple_agent_demo()
        elif choice == "3":
            await interactive_demo()
        else:
            print("❌ Invalid choice. Running Single Agent Demo by default...")
            await single_agent_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️ Interrupted by user.")
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())