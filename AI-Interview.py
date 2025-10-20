import streamlit as st
import autogen
from autogen.agentchat import AssistantAgent, UserProxyAgent
from typing import Dict, Optional
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Interview System - AutoGen",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .interviewer-msg {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2196F3;
    }
    .candidate-msg {
        background-color: #f3e5f5;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #9C27B0;
    }
    .coach-msg {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #FF9800;
    }
    .scorer-msg {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.conversation_history = []
    st.session_state.interview_active = False
    st.session_state.question_count = 0
    st.session_state.total_score = 0
    st.session_state.scores_list = []
    st.session_state.agents_ready = False
    st.session_state.current_question = ""
    st.session_state.waiting_for_answer = False

def create_agents(api_key: str, model: str, job_role: str, experience_level: str):
    """Initialize all AutoGen agents with specific roles"""
    
    config_list = [{
        "model": model,
        "api_key": api_key,
        "temperature": 0.7,
    }]
    
    llm_config = {
        "config_list": config_list,
        "timeout": 120,
    }
    
    # Interviewer Agent - Asks questions
    interviewer = autogen.AssistantAgent(
        name="Interviewer",
        system_message=f"""You are a professional interviewer conducting an interview for a {job_role} position ({experience_level} level).

Your role:
- Ask ONE clear, relevant technical or behavioral question at a time
- Tailor questions to {job_role} role and {experience_level} experience level
- Be professional, friendly, and encouraging
- After candidate answers, acknowledge briefly before the next question
- Keep questions concise and focused

Ask your first question now.""",
        llm_config=llm_config,
    )
    
    # Coach Agent - Analyzes and gives suggestions
    coach = autogen.AssistantAgent(
        name="Coach",
        system_message=f"""You are an expert interview coach analyzing candidate responses for a {job_role} position.

Your role:
- Analyze the candidate's answer in detail
- Identify strengths in their response
- Provide 3-4 specific, actionable suggestions on how to improve and succeed
- Focus on: content quality, structure, communication style, technical accuracy
- Be encouraging but honest
- Give practical tips they can apply to the next answer

Format:
✅ Strengths: [what they did well]
💡 Suggestions for Success:
1. [Specific actionable tip]
2. [Specific actionable tip]
3. [Specific actionable tip]""",
        llm_config=llm_config,
    )
    
    # Scorer Agent - Evaluates and scores
    scorer = autogen.AssistantAgent(
        name="Scorer",
        system_message=f"""You are an objective interview evaluator scoring responses for a {job_role} position.

Your role:
- Review the candidate's answer against the question asked
- Score from 1-10 based on:
  * Technical Accuracy (if applicable): 30%
  * Clarity and Structure: 25%
  * Completeness: 25%
  * Relevance: 20%
- Provide brief justification for the score
- Be fair and consistent

Format:
📊 SCORE: [X]/10

Breakdown:
- [Criterion 1]: [score/points]
- [Criterion 2]: [score/points]
- [Criterion 3]: [score/points]

Justification: [2-3 sentences explaining the score]""",
        llm_config=llm_config,
    )
    
    return interviewer, coach, scorer

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Configuration
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
    model = st.selectbox(
        "Model",
        ["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"],
        index=0
    )
    
    st.divider()
    
    # Interview Settings
    st.subheader("📋 Interview Settings")
    job_role = st.text_input("Job Role", value="Software Engineer", help="The position being interviewed for")
    experience_level = st.selectbox(
        "Experience Level",
        ["Junior", "Mid-Level", "Senior", "Lead/Principal"]
    )
    num_questions = st.number_input("Number of Questions", min_value=1, max_value=15, value=5)
    
    st.divider()
    
    # Control Buttons
    if not st.session_state.interview_active:
        if st.button("🚀 Start Interview", type="primary"):
            if not api_key:
                st.error("⚠️ Please enter your OpenAI API key!")
            else:
                try:
                    with st.spinner("Initializing AI agents..."):
                        interviewer, coach, scorer = create_agents(api_key, model, job_role, experience_level)
                        st.session_state.interviewer = interviewer
                        st.session_state.coach = coach
                        st.session_state.scorer = scorer
                        st.session_state.agents_ready = True
                        st.session_state.interview_active = True
                        st.session_state.question_count = 0
                        st.session_state.conversation_history = []
                        st.session_state.scores_list = []
                        st.session_state.total_score = 0
                        
                        # Generate first question
                        first_question = interviewer.generate_reply(
                            messages=[{"role": "user", "content": "Ask the first interview question."}]
                        )
                        
                        st.session_state.current_question = first_question
                        st.session_state.conversation_history.append({
                            "role": "Interviewer",
                            "content": first_question,
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                        st.session_state.waiting_for_answer = True
                        st.session_state.question_count = 1
                        
                    st.success("✅ Interview started!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error initializing agents: {str(e)}")
    else:
        if st.button("🛑 End Interview", type="secondary"):
            st.session_state.interview_active = False
            st.session_state.waiting_for_answer = False
            st.rerun()
    
    if st.button("🔄 Reset All"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.divider()
    
    # Interview Stats
    if st.session_state.interview_active:
        st.subheader("📊 Interview Stats")
        st.metric("Questions Asked", f"{st.session_state.question_count}/{num_questions}")
        
        if st.session_state.scores_list:
            avg_score = sum(st.session_state.scores_list) / len(st.session_state.scores_list)
            st.metric("Average Score", f"{avg_score:.1f}/10")
            
            # Score trend
            st.line_chart(st.session_state.scores_list)

# Main Content Area
st.title("🎯 AI Interview System")
st.markdown(f"**Multi-Agent Interview Assistant powered by AutoGen**")

if not st.session_state.interview_active:
    st.info("👈 Configure your interview settings in the sidebar and click 'Start Interview' to begin!")
    
    # Feature overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 👔 Interviewer")
        st.write("Asks relevant questions tailored to your role")
    
    with col2:
        st.markdown("### 👤 Candidate")
        st.write("You provide answers through the chat")
    
    with col3:
        st.markdown("### 💡 Coach")
        st.write("Analyzes responses and suggests improvements")
    
    with col4:
        st.markdown("### 📊 Scorer")
        st.write("Evaluates answers and provides scores")

else:
    # Display conversation history
    st.subheader("💬 Interview Conversation")
    
    conversation_container = st.container(height=400)
    
    with conversation_container:
        for msg in st.session_state.conversation_history:
            role = msg["role"]
            content = msg["content"]
            timestamp = msg.get("timestamp", "")
            
            if role == "Interviewer":
                st.markdown(f"""
                <div class="interviewer-msg">
                    <b>👔 Interviewer</b> <span style="color: #666; font-size: 0.8em;">{timestamp}</span><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)
                
            elif role == "Candidate":
                st.markdown(f"""
                <div class="candidate-msg">
                    <b>👤 You (Candidate)</b> <span style="color: #666; font-size: 0.8em;">{timestamp}</span><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)
                
            elif role == "Coach":
                st.markdown(f"""
                <div class="coach-msg">
                    <b>💡 Coach Analysis</b> <span style="color: #666; font-size: 0.8em;">{timestamp}</span><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)
                
            elif role == "Scorer":
                st.markdown(f"""
                <div class="scorer-msg">
                    <b>📊 Score Evaluation</b> <span style="color: #666; font-size: 0.8em;">{timestamp}</span><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)
    
    # Input area for candidate response
    st.divider()
    
    if st.session_state.waiting_for_answer and st.session_state.question_count <= num_questions:
        st.subheader("✍️ Your Answer")
        
        with st.form(key="answer_form", clear_on_submit=True):
            user_answer = st.text_area(
                "Type your answer here:",
                height=150,
                placeholder="Provide a detailed answer to the question above...",
                key=f"answer_input_{st.session_state.question_count}"
            )
            
            submit_button = st.form_submit_button("Submit Answer", type="primary")
            
            if submit_button and user_answer.strip():
                # Add candidate answer to history
                st.session_state.conversation_history.append({
                    "role": "Candidate",
                    "content": user_answer,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                try:
                    with st.spinner("🤔 AI agents are analyzing your response..."):
                        # Coach analyzes the response
                        coach_feedback = st.session_state.coach.generate_reply(
                            messages=[{
                                "role": "user",
                                "content": f"Question: {st.session_state.current_question}\n\nCandidate's Answer: {user_answer}\n\nProvide detailed coaching feedback with suggestions for success."
                            }]
                        )
                        
                        st.session_state.conversation_history.append({
                            "role": "Coach",
                            "content": coach_feedback,
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                        
                        # Scorer evaluates the response
                        score_feedback = st.session_state.scorer.generate_reply(
                            messages=[{
                                "role": "user",
                                "content": f"Question: {st.session_state.current_question}\n\nCandidate's Answer: {user_answer}\n\nProvide a detailed score and evaluation."
                            }]
                        )
                        
                        st.session_state.conversation_history.append({
                            "role": "Scorer",
                            "content": score_feedback,
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                        
                        # Extract score
                        try:
                            score_line = [line for line in score_feedback.split('\n') if 'SCORE:' in line.upper()][0]
                            score = float(score_line.split('/')[0].split(':')[-1].strip())
                            st.session_state.scores_list.append(score)
                            st.session_state.total_score += score
                        except:
                            st.session_state.scores_list.append(7.0)
                            st.session_state.total_score += 7.0
                        
                        # Generate next question if interview not complete
                        if st.session_state.question_count < num_questions:
                            next_question = st.session_state.interviewer.generate_reply(
                                messages=[{
                                    "role": "user",
                                    "content": f"Based on the previous answer, ask the next relevant interview question. Previous Q&A context:\n\nQ: {st.session_state.current_question}\nA: {user_answer}"
                                }]
                            )
                            
                            st.session_state.current_question = next_question
                            st.session_state.conversation_history.append({
                                "role": "Interviewer",
                                "content": next_question,
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            st.session_state.question_count += 1
                        else:
                            st.session_state.waiting_for_answer = False
                            st.session_state.interview_active = False
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing response: {str(e)}")
            
            elif submit_button:
                st.warning("⚠️ Please provide an answer before submitting.")
    
    # Interview completion
    if not st.session_state.waiting_for_answer and st.session_state.question_count >= num_questions:
        st.success("🎉 Interview Completed!")
        
        st.divider()
        
        # Final Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Questions", st.session_state.question_count)
        
        with col2:
            avg_score = st.session_state.total_score / len(st.session_state.scores_list) if st.session_state.scores_list else 0
            st.metric("Average Score", f"{avg_score:.2f}/10")
        
        with col3:
            performance = "Excellent" if avg_score >= 8 else "Good" if avg_score >= 6 else "Needs Improvement"
            st.metric("Overall Performance", performance)
        
        # Score breakdown
        st.subheader("📈 Score Breakdown")
        score_data = {f"Q{i+1}": score for i, score in enumerate(st.session_state.scores_list)}
        st.bar_chart(score_data)
        
        # Download Report
        st.divider()
        st.subheader("📥 Download Interview Report")
        
        report = {
            "interview_details": {
                "job_role": job_role,
                "experience_level": experience_level,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_questions": st.session_state.question_count
            },
            "performance": {
                "scores": st.session_state.scores_list,
                "average_score": avg_score,
                "total_score": st.session_state.total_score
            },
            "conversation": st.session_state.conversation_history
        }
        
        report_json = json.dumps(report, indent=2)
        
        st.download_button(
            label="📄 Download Full Report (JSON)",
            data=report_json,
            file_name=f"interview_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

# Footer
st.divider()
st.caption("🤖 AI Interview System | Built with AutoGen & Streamlit | Multi-Agent Framework")