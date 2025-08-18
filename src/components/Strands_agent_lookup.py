import json
from strands import Agent
from strands.models import BedrockModel
import boto3

# System prompt template with JSON placeholder
SYSTEM_PROMPT_TEMPLATE = """
You are a department matcher for a wealth management chatbot. Based on the provided LLM response, identify the most relevant department by matching the response content to the 'High_Level_Tasks' (or similar fields) listed for each department in the provided JSON data.

LLM Response:
{llm_response}

Departments data:
{departments_json}

Steps to reason:
1. Analyze the LLM response for key actions or topics (e.g., mention of 'selling a bond' relates to 'New Issue Offerings' or 'Municipal Trading').
2. Compare to the 'High_Level_Tasks' (or equivalent fields) of each department.
3. Select the best matching department (or 'None' if unclear).
4. If a match is found, return the contact details in JSON format overwriting the prompt template fields: 
   {{"department": "name", "manager_name": "Manager Name", "manager_phone": "Manager Phone", "supervisor_name": "Supervisor Name", "supervisor_phone": "Supervisor Phone", "support_line": "Operations_Support_Line"}}.
   Use the 'Leadership_Team_Contact' structure to extract these values.
5. If no match is found, return {{"department": "None"}} (generic contact will be applied elsewhere).
"""

# Create the Strands agent with error handling and model fallback
def create_department_agent(departments_data, llm_response):
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(departments_json=json.dumps(departments_data, indent=2), llm_response=llm_response)
    
    # List of fallback models based on available options
    model_options = [
        "anthropic.claude-3-sonnet-20240229-v1:0",  # Preferred from available models
        "anthropic.claude-3-haiku-20240307-v1:0",   # Secondary option
        "anthropic.claude-v2:1",                    # Older stable option
        "anthropic.claude-v2"                       # Last resort
    ]
    
    bedrock_model = None
    bedrock_client = boto3.client('bedrock-runtime', region_name="us-east-1")
    available_models = [m['modelId'] for m in bedrock_client.list_foundation_models()['modelSummaries']]
    print(f"Available Bedrock models: {available_models}")
    
    for model_id in model_options:
        if model_id in available_models:
            try:
                bedrock_model = BedrockModel(model_id=model_id, region_name="us-east-1")
                print(f"Successfully initialized Bedrock model: {model_id}")
                break
            except Exception as e:
                print(f"Failed to initialize model {model_id}: {str(e)}. Trying next option...")
        else:
            print(f"Model {model_id} not available in current region.")
    
    if bedrock_model is None:
        raise ValueError(f"No valid Bedrock models available from {model_options}. Please check your Bedrock configuration and model access. Available models: {available_models}")

    return Agent(
        system_prompt=system_prompt,
        model=bedrock_model
    )

# Function to get department contact based on LLM response and metadata
def get_department_contact_from_response(llm_response: str, metadata_text: dict):
    agent = create_department_agent(metadata_text, llm_response)
    response = agent(llm_response)  # Use the response as input to maintain consistency
    try:
        # Attempt to extract the raw string from AgentResult
        response_text = getattr(response, 'content', str(response))  # Fallback to str if content not available
        return json.loads(response_text)  # Parse the JSON string
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)}. Returning default response.")
        return {"department": "None"}
    except AttributeError as e:
        print(f"Attribute error accessing response: {str(e)}. Returning default response.")
        return {"department": "None"}

# Example usage (for testing)
if __name__ == "__main__":
    sample_metadata = {
        "Fixed_Income_Trading": {
            "Leadership_Team_Contact": {
                "High_Level_Tasks": ["Municipal Trading", "New Issue Offerings"],
                "Manager": {"Name": "Rob Jones", "Phone": "(111) 222-7340"},
                "Team_Lead": {"Name": "Samantha", "Phone": "(111) 222-7305"},
                "Support_Line": "855-756-1111, Option 1, Option 1"
            }
        }
    }
    sample_response = "To sell a bond on Bond Beacon, follow these steps... involving trading."
    print(get_department_contact_from_response(sample_response, sample_metadata))