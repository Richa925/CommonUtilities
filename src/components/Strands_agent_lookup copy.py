import json
from strands import Agent
from strands.models import BedrockModel

# Load departments JSON dynamically
def load_departments_json(file_path='opsmetadta_FINAL.json'):
    with open(file_path, 'r') as f:
        return json.load(f)

# System prompt template with JSON placeholder
SYSTEM_PROMPT_TEMPLATE = """
You are a department matcher for a operations management chatbot. Based on the user's query, identify the most relevant department by matching the query to the 'High_Level_Tasks' listed for each department in the provided JSON data.

Departments data:
{departments_json}

Steps to reason:
1. Analyze the query for key actions or topics (e.g., "buying a bond" relates to "New Issue Offerings").
2. Compare to the 'High_Level_Tasks' field of each department.
3. Select the best matching department (or 'None' if unclear).
4. Return ONLY the contact details in JSON format: {{"department": "name", "manager_name": "name", "manager_phone": "number", "support_line": "number"}}.
If no match, return {{"department": "None"}}.
"""

# Create the Strands agent
def create_department_agent():
    departments_data = load_departments_json()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(departments_json=json.dumps(departments_data, indent=2))
    
    bedrock_model = BedrockModel(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",  # Adjust to your preferred Bedrock model
        region_name="us-east-1"  # Adjust to your region
    )
    
    return Agent(
        system_prompt=system_prompt,
        model=bedrock_model
    )

# Function to get department contact based on query
def get_department_contact(query: str):
    agent = create_department_agent()
    response = agent(query)
    try:
        return json.loads(response)  # Parse the agent's JSON output
    except json.JSONDecodeError:
        return {"department": "Error", "message": "Failed to parse contact info"}

# Example usage (for testing)
if __name__ == "__main__":
    print(get_department_contact("How do I buy a bond?"))