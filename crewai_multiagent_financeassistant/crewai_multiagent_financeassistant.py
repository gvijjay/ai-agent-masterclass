import streamlit as st
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai_api_key = st.secrets["OPENAI_API_KEY"]
if not openai_api_key:
    st.error("OPENAI_API_KEY not found in .env file!")
    st.stop()

#os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"  # Or your preferred model

# Define Agents with backstories
budget_analyst = Agent(
    role="Budget Analyst",
    goal="Provide a clear budget breakdown.",
    backstory="A meticulous financial expert who ensures optimal budgeting.",
    verbose=True
)

spending_advisor = Agent(
    role="Spending Advisor",
    goal="Identify excessive spending and suggest cuts.",
    backstory="A frugal specialist who detects unnecessary spending.",
    verbose=True
)

investment_advisor = Agent(
    role="Investment Advisor",
    goal="Suggest 3 low-risk investments.",
    backstory="A seasoned investor with a cautious, growth-focused approach.",
    verbose=True
)

savings_planner = Agent(
    role="Savings Planner",
    goal="Recommend one savings option.",
    backstory="A financial planner who prioritizes secure saving strategies.",
    verbose=True
)

report_generator = Agent(
    role="Report Generator",
    goal="Compile a concise financial plan.",
    backstory="A skilled financial writer who presents data clearly.",
    verbose=True
)

# Streamlit UI
st.title("Personal Finance Assistant")

# User Inputs
income = st.number_input("Monthly Income ($)", min_value=0.0, step=100.0, value=3000.0)
st.write("Enter your monthly expenses (e.g., 'Rent: 1000'):")
expense_inputs = [st.text_input(f"Expense {i+1}") for i in range(4)]
savings_goal = st.number_input("Savings Goal ($)", min_value=0.0, step=50.0, value=500.0)
risk_tolerance = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"], index=0)
max_investment = st.number_input("Max Investment Amount ($)", min_value=0.0, step=50.0, value=200.0)

# Process Expenses
expenses = {}
total_expenses = 0
for exp in expense_inputs:
    if exp:
        try:
            category, amount = exp.split(":")
            amount = float(amount.strip())
            expenses[category.strip()] = amount
            total_expenses += amount
        except ValueError:
            st.error(f"Invalid format: '{exp}'. Use 'Category: Amount'.")

# Run Crew Button
if st.button("Generate Financial Plan"):
    if income < total_expenses + savings_goal:
        st.error("Income must cover expenses and savings goal!")
    else:
        # Define Tasks with expected outputs
        budget_task = Task(
            description=f"Break down: income ${income}, expenses {expenses}, savings goal ${savings_goal}.",
            expected_output="A clear budget breakdown.",
            agent=budget_analyst
        )

        spending_task = Task(
            description=f"Check {expenses} against ${income}. Flag excessive spending.",
            expected_output="List of excessive spending with reduction tips.",
            agent=spending_advisor
        )

        investment_task = Task(
            description=f"Suggest 3 investments under ${max_investment} for {risk_tolerance} risk.",
            expected_output="3 investment suggestions with explanations.",
            agent=investment_advisor
        )

        savings_task = Task(
            description=f"Recommend one savings option for ${savings_goal}.",
            expected_output="A savings option with reasoning.",
            agent=savings_planner
        )

        report_task = Task(
            description="Combine all financial details into a concise plan.",
            expected_output="A well-structured financial summary.",
            agent=report_generator
        )

        # Assemble and Run Crew
        finance_crew = Crew(
            agents=[budget_analyst, spending_advisor, investment_advisor, savings_planner, report_generator],
            tasks=[budget_task, spending_task, investment_task, savings_task, report_task],
            process=Process.sequential
        )

        with st.spinner("Generating your financial plan..."):
            result = finance_crew.kickoff()

        # Display Results
        st.subheader("Your Financial Plan")

        try:
            st.write("### Budget Breakdown")
            st.text(result.tasks_output[0].raw if hasattr(result.tasks_output[0], 'raw') else "Budget breakdown unavailable.")

            st.write("### Spending Analysis")
            st.text(result.tasks_output[1].raw if hasattr(result.tasks_output[1], 'raw') else "No excessive spending detected.")

            st.write("### Investment Plan")
            st.text(result.tasks_output[2].raw if hasattr(result.tasks_output[2], 'raw') else "No investment suggestions available.")

            st.write("### Savings Plan")
            st.text(result.tasks_output[3].raw if hasattr(result.tasks_output[3], 'raw') else "No savings option recommended.")

            st.write("### Summary")
            st.text(result.tasks_output[4].raw if hasattr(result.tasks_output[4], 'raw') else "Summary unavailable.")

        except IndexError:
            st.error("An error occurred while processing the financial plan. Please try again.")
        except AttributeError:
            st.error("Task output format has changed. Please check the latest CrewAI documentation for the correct attribute.")