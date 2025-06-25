import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db, generate_random_question_from_vectordb
# from langchain_community.cross_encoders import HuggingFaceCrossEncoder # For similarity comparison
from sentence_transformers import CrossEncoder

# --- Mock credential store ---
AUTHORIZED_USERS = {
    "admin": "password123",
    # Add more users if needed
}

# --- Session State Initialization ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "question" not in st.session_state: # This is for the main Q&A
    st.session_state.question = ""
if "quiz_active" not in st.session_state:
    st.session_state.quiz_active = False
if "current_quiz_question" not in st.session_state:
    st.session_state.current_quiz_question = ""
if "correct_quiz_answer" not in st.session_state:
    st.session_state.correct_quiz_answer = ""
if "user_quiz_answer" not in st.session_state:
    st.session_state.user_quiz_answer = ""

# --- Title ---
st.title("Smart Quiz")

# --- Logout Option ---
if st.session_state.authenticated:
    st.write(f"Logged in as: **{st.session_state.username}**")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.rerun()

# --- Login Section ---
with st.expander("🔐 Login"):
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")
    if st.button("Login"):
        if username in AUTHORIZED_USERS and AUTHORIZED_USERS[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid credentials")

# --- Main Q&A Interface ---
st.header("General Q&A")
st.text_input("Ask a question:", key="question_input", value=st.session_state.question)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Submit Question", key="submit_main_qa"): # Changed key for uniqueness
        question = st.session_state.question_input
        if question.strip():
            st.session_state.question = question
            with st.spinner("Finding answer..."):
                chain = get_qa_chain()
                response = chain(question)
            st.header("Answer")
            st.write(response["result"])
with col2:
    if st.button("Clear Q&A", key="clear_main_qa"): # Changed key for uniqueness
        st.session_state.question = ""
#        st.session_state.question_input = "" # Clear the text_input as well
        st.rerun()

# --- New Quiz Section ---
st.markdown("---")
st.header("Quiz Yourself!")

if not st.session_state.quiz_active:
    if st.button("Start New Quiz"):
        # Generate a random question
        with st.spinner("Generating question..."):
            random_question_text = generate_random_question_from_vectordb()
            
            # Get the "correct" answer from the RAG chain for comparison later
            qa_chain = get_qa_chain()
            correct_answer_response = qa_chain({"query": random_question_text})
            correct_answer_text = correct_answer_response["result"]

        st.session_state.current_quiz_question = random_question_text
        st.session_state.correct_quiz_answer = correct_answer_text
        st.session_state.quiz_active = True
        st.session_state.user_quiz_answer = "" # Reset user's previous answer
        st.rerun()
else:
    st.write("---")
    st.subheader("Your Quiz Question:")
    st.info(st.session_state.current_quiz_question)

    st.text_area("Your Answer:", key="user_quiz_answer_input", value=st.session_state.user_quiz_answer, height=150)

    col_quiz1, col_quiz2 = st.columns([1,1])
    with col_quiz1:
        if st.button("Submit Quiz Answer"):
            st.session_state.user_quiz_answer = st.session_state.user_quiz_answer_input
            if not st.session_state.user_quiz_answer.strip():
                st.warning("Please type your answer before submitting.")
            else:
                # Compare answers
                # model = HuggingFaceCrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                
                # We compare user's answer against the RAG's answer
                # A higher score means closer semantic meaning
                similarity_score = model.predict([
                    st.session_state.correct_quiz_answer,
                    st.session_state.user_quiz_answer
                ])

                st.subheader("Quiz Result:")
                st.write(f"**Your Answer:** {st.session_state.user_quiz_answer}")
                st.write(f"**Correct Answer (from Knowledgebase):** {st.session_state.correct_quiz_answer}")
                st.write(f"**Similarity Score:** {similarity_score:.2f} (closer to 1 means more similar)")

                if similarity_score >= 0.7: # Threshold can be adjusted
                    st.success("Great job! Your answer is very similar to the knowledgebase's information.")
                elif similarity_score >= 0.4:
                    st.warning("Your answer has some similarities, but could be improved.")
                else:
                    st.error("Your answer is quite different from the knowledgebase's information.")
    with col_quiz2:
        if st.button("End Quiz"):
            st.session_state.quiz_active = False
            st.session_state.current_quiz_question = ""
            st.session_state.correct_quiz_answer = ""
            st.session_state.user_quiz_answer = ""
            st.rerun()


# --- Admin Controls ---
def show_admin_controls():
    if st.session_state.authenticated and st.session_state.username == "admin":
        st.markdown("---")
        st.subheader("Admin Controls")
        if st.button("📚 Create Knowledgebase"):
            with st.spinner("Creating/Updating Knowledgebase... This may take a moment."):
                create_vector_db()
            st.success("Knowledgebase created/updated!")

show_admin_controls()