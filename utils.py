def build_prompt(role, goal, examples, user_input):
    return f"""
You are {role}.
Your goal: {goal}.
Here are examples:\n{examples}
Now respond to: {user_input}
"""
