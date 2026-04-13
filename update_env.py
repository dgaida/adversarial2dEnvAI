import sys

file_path = "src/custom_grid_env/env.py"
with open(file_path, "r") as f:
    content = f.read()

search_text = """        preserved_info = {}
        if "cnn_prediction" in info:
            preserved_info["cnn_prediction"] = info["cnn_prediction"]
        if "cnn_probs" in info:
            preserved_info["cnn_probs"] = info["cnn_probs"]"""

replace_text = """        preserved_info = {}
        keys_to_preserve = [
            "cnn_prediction",
            "cnn_probs",
            "estimated_pos",
            "color_measurement",
            "intended_action",
            "actual_action",
            "slipped",
            "particles",
        ]
        for key in keys_to_preserve:
            if key in info:
                preserved_info[key] = info[key]"""

new_content = content.replace(search_text, replace_text)
with open(file_path, "w") as f:
    f.write(new_content)
