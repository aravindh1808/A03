import os
from crewai import Agent, Task, Crew

def main():
    # 1) Check that the key is visible to Python
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is NOT visible in this process.")
        print("Make sure it is set as a system/user env var and restart your terminal/IDE.")
        return
    else:
        print("OPENAI_API_KEY is set. Testing CrewAI + LLM...\n")

    # 2) Define a simple agent
    test_agent = Agent(
        role="Test LLM Agent",
        goal="Confirm that CrewAI and the LLM are working.",
        backstory=(
            "You are a tiny diagnostic agent whose only job is to say a short "
            "confirmation message when things work."
        ),
        verbose=True,
    )

    # 3) Define a simple task (note expected_output is REQUIRED)
    test_task = Task(
        description="Say one short sentence confirming that CrewAI is working correctly.",
        expected_output="A single short confirmation sentence.",
        agent=test_agent,
    )

    # 4) Create a crew and run it
    crew = Crew(
        agents=[test_agent],
        tasks=[test_task],
        verbose=True,
    )

    result = crew.kickoff()
    print("\n--- Result from LLM ---")
    print(result)

if __name__ == "__main__":
    main()
