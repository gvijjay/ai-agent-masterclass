import openai
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from the .env file
load_dotenv()

api_key = st.secrets["OPENAI_API_KEY"]
# Get the OpenAI API key from environment variables
openai.api_key = api_key

# Step 1: Reasoning (Understanding the task and planning)
def reasoning_about_task(topic):
    reasoning_prompt = f"Given the topic '{topic}', what are the key points that should be included in a coherent paragraph? Make sure to consider structure, clarity, and relevance."
    
    # Using GPT-4o-mini to reason about the topic (use ChatCompletion endpoint)
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # Use GPT-4o-mini model
        messages=[{"role": "user", "content": reasoning_prompt}],
        max_tokens=700,
        temperature=0.7
    )
    
    reasoning_output = response['choices'][0]['message']['content'].strip()
    return reasoning_output

# Step 2: Acting (Writing the paragraph based on reasoning)
def generate_paragraph(topic, reasoning_output):
    act_prompt = f"Write a well-structured paragraph on the topic '{topic}' using these key points: {reasoning_output}. Be clear and coherent."
    
    # Using GPT-4o-mini to generate a paragraph based on reasoning (use ChatCompletion endpoint)
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  
        messages=[{"role": "user", "content": act_prompt}],
        max_tokens=700,
        temperature=0.7
    )
    
    paragraph = response['choices'][0]['message']['content'].strip()
    return paragraph

# Step 3: Reflection (Review the generated paragraph and reflect)
def reflect_on_paragraph(paragraph):
    reflection_prompt = f"Review the following paragraph: '{paragraph}'. Does it clearly address the topic? Are there areas for improvement, such as clarity, detail, or structure?"
    
    # Using GPT-4o-mini to reflect and critique the paragraph (use ChatCompletion endpoint)
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  
        messages=[{"role": "user", "content": reflection_prompt}],
        max_tokens=700,
        temperature=0.7
    )
    
    reflection_output = response['choices'][0]['message']['content'].strip()
    return reflection_output

# Step 4: Iteration (Refine the paragraph based on reflection)
def refine_paragraph(paragraph, reflection_output):
    refine_prompt = f"Refine the following paragraph based on the feedback: '{reflection_output}'. Here's the paragraph: '{paragraph}'."
    
    # Using GPT-4o-mini to refine the paragraph (use ChatCompletion endpoint)
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  
        messages=[{"role": "user", "content": refine_prompt}],
        max_tokens=700,
        temperature=0.7
    )
    
    refined_paragraph = response['choices'][0]['message']['content'].strip()
    return refined_paragraph

# Streamlit UI
def main():
    st.title("AI Agent Scratch paragraph generator")
    st.write("Ask a question or enter a topic, and the AI will generate a reasoned and refined response based on the topic.")

    # User input for the topic or question
    user_input = st.text_input("Enter your topic/question:")

    if user_input:
        with st.spinner("Thinking..."):
            # Step 1: Reason about the task
            reasoning_output = reasoning_about_task(user_input)
            st.write("### Reasoning Output:")
            st.write(reasoning_output)

            # Step 2: Generate the paragraph
            paragraph = generate_paragraph(user_input, reasoning_output)
            st.write("### Generated Paragraph:")
            st.write(paragraph)

            # Step 3: Reflect on the generated paragraph
            reflection_output = reflect_on_paragraph(paragraph)
            st.write("### Reflection Output:")
            st.write(reflection_output)

            # Step 4: Refine the paragraph based on reflection
            refined_paragraph = refine_paragraph(paragraph, reflection_output)
            st.write("### Refined Paragraph:")
            st.write(refined_paragraph)

if __name__ == "__main__":
    main()
