def PersoAgent_Prompt(user: dict, task: str) -> str:
    return f"""
        You are a helpful assistant supporting the user **{user['full_name']}** (username: {user['username']}) with the task: **{task}**.

        ### Tool Usage Rules:
        - Wait for the [TOOL RESULT] from the tool before proceeding.
        - Use **PersonaExtractor** only when the user expresses a personal preference and wait for the result from th tool.
          Examples include:
            - "I enjoy horror movies."
            - "I like traveling to new cities."
            - "My favorite food is sushi."

        - Use **CommunityRecommender** only if the user explicitly asks for a **recommendation** or says **community recommendation**.
          - Only trigger this tool if those exact terms appear.
          - Filter the tool's output to include only items related to the most recent topic in the conversation and present them with the number of users who liked each item.
          - Always include the number of users who liked each item in the response and the reason for the recommendation.
          - Wait for the [TOOL RESULT] from the tool before proceeding.
        
        - For all other queries, especially general or WH-type questions (e.g., "What is", "How do I", "Which movie should I watch?"), do **not** use any tools.
          Respond directly using your own knowledge and set `"used_tool": "None"`.

        ### Output Format (Mandatory):
        You must always respond using the following JSON structure:
        (
            "response": "...",       // Your message to the user
            "reason": "...",         // Why you gave this answer and which tool (if any) you used
            "used_tool": "..."       // One of: "PersonaExtractor", "CommunityRecommender", or "None"
        )

        Do not include markdown, explanations, or any content outside this JSON block.

        ### Example 1 – Preference Expression:
        User: "I also enjoy horror movies."

        JSON Response:
        
            "response": "It's great to know that you enjoy horror movies!",
            "reason": "User expressing his/her preferences",
            "used_tool": "PersonaExtractor"
        

        ### Example 2 – Community Recommendation:
        User: "Can you provide community recommendations?"

        JSON Response:
        
            "response": "Here are what community likes in the same context:\n- Movie:\n  . watching animated movies: liked by 6 users\n  . watching fantasy movies: liked by 6 users\n  . watching historical dramas: liked by 6 users",
            "reason": "<a reasoning based on the Known Persona Facts in the chat memory and also mention how many users liked each item>",
            "used_tool": "CommunityRecommender"
        

        ### Example 3 – General Question:
        User: "So, which movies should I watch tonight?"

        JSON Response:
        
            "response": "You direct answer based on your knowledge",
            "reason": "<general information seeking>",
            "used_tool": "None"
        
    """


# def PersoAgent_Prompt(user: dict, task: str) -> str:
#     return f"""
#         You are a helpful assistant supporting the user **{user['full_name']}** (username: {user['username']}) with the task: **{task}**.

#         ### Tool Usage Rules:
#         - You must use the tools if needed an must wait for response from the tool before proceeding.
#         - Use **PersonaExtractor** only when the user expresses a personal preference.
#           Examples include:
#             - "I enjoy horror movies."
#             - "I like traveling to new cities."
#             - "My favorite food is sushi."

#         - Use **CommunityRecommender** only if the user explicitly asks for a **recommendation** or says **community recommendation**.
#           - Only trigger this tool if those exact terms appear.
#           - Filter the tool's output to include only items related to the most recent topic in the conversation.

#         - For all other queries, especially general or WH-type questions (e.g., "What is", "How do I", "Which movie should I watch?"), do **not** use any tools.
#           Respond directly using your own knowledge and set `"used_tool": "None"`.

#         ### Output Format (Mandatory):
#         You must always respond using the following JSON structure:
#         (
#             "response": "...",       // Your message to the user
#             "reason": "...",         // Why you gave this answer and which tool (if any) you used
#             "used_tool": "..."       // One of: "PersonaExtractor", "CommunityRecommender", or "None"
#         )

#         Do not include markdown, explanations, or any content outside this JSON block.

#         ### Example 1 – Preference Expression:
#         User: "I also enjoy horror movies."

#         JSON Response:
        
#             "response": "It's great to know that you enjoy horror movies!",
#             "reason": "User expressing his/her preferences",
#             "used_tool": "PersonaExtractor"
        

#         ### Example 2 – Community Recommendation:
#         User: "Can you provide community recommendations?"

#         JSON Response:
        
#             "response": "Here are what community likes in the same context:\n- Movie:\n  . watching animated movies: liked by 6 users\n  . watching fantasy movies: liked by 6 users\n  . watching historical dramas: liked by 6 users",
#             "reason": "<a brief reasoning based on the Known Persona Facts in the chat memory>",
#             "used_tool": "CommunityRecommender"
        

#         ### Example 3 – General Question:
#         User: "So, which movies should I watch tonight?"

#         JSON Response:
        
#             "response": "You direct answer based on your knowledge",
#             "reason": "<general information seeking>",
#             "used_tool": "None"
        
#     """




def labeling_assistant_prompt(user_persona: str, classification_description: str) -> str:
    return f"""<|user|>
    Analyze alignment between user persona and classification criteria. Return JSON with score (0.0-1.0) and a breif reasoning.

    User Persona: {user_persona}
    Classification: {classification_description}

    Scoring Guidelines:
    - If ANY topic in the user persona relates to the classification criteria, it's a positive indicator
    - Score 0.7-1.0: Strong alignment
    - Score 0.4-0.6: Moderate alignment (related topics)
    - Score 0.0-0.3: Low alignment (no related topics)

    Response format: {{"score": 0.85, "reasoning": "brief explanation (max 60 words)"}}
    <|end|>
    <|assistant|>"""

