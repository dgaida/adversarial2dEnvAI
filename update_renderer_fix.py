import sys

file_path = "src/custom_grid_env/renderer.py"
with open(file_path, "r") as f:
    content = f.read()

# Fix the syntax error in turn_name
content = content.replace('turn_name = "Agent"s Turn"', "turn_name = \"Agent's Turn\"")
content = content.replace('else "Ghost"s Turn"', "else \"Ghost's Turn\"")

with open(file_path, "w") as f:
    f.write(content)
