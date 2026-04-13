import sys

file_path = "src/custom_grid_env/renderer.py"
with open(file_path, "r") as f:
    lines = f.readlines()

start_line = -1
end_line = -1
for i, line in enumerate(lines):
    if "def _draw_info_panel(" in line:
        start_line = i
    if start_line != -1 and "def render(" in line:
        end_line = i
        break

if start_line == -1 or end_line == -1:
    print(f"Could not find _draw_info_panel. start={start_line}, end={end_line}")
    sys.exit(1)

new_method = """    def _draw_info_panel(
        self,
        agent_pos: List[int],
        ghost_pos: List[int],
        step_count: int,
        current_turn: int,
        grid: np.ndarray,
        info: Dict[str, Any],
    ):
        \"\"\"Draws the information panel at the bottom.\"\"\"
        panel_y = self.rows * self.cell_size
        pygame.draw.rect(
            self.screen, self.colors["dark_gray"], (0, panel_y, self.window_width, 185)
        )

        # Column offsets
        col1_x = 10
        col2_x = 180
        col3_x = 330

        # Row 1: Step and Turn
        step_text = self.font.render(f"Step: {step_count}", True, self.colors["white"])
        self.screen.blit(step_text, (col1_x, panel_y + 10))

        turn_name = "Agent\"s Turn" if current_turn == 0 else "Ghost\"s Turn"
        turn_color = self.colors["yellow"] if current_turn == 0 else self.colors["cyan"]
        turn_text = self.font.render(turn_name, True, turn_color)
        self.screen.blit(turn_text, (col2_x, panel_y + 10))

        # CNN Prediction (Row 1, Col 3)
        prediction = info.get("cnn_prediction")
        if prediction:
            class_name, prob = prediction
            pred_text = self.small_font.render(
                f"CNN: {class_name} ({prob*100:.1f}%)",
                True,
                self.colors["orange"],
            )
            self.screen.blit(pred_text, (col3_x, panel_y + 10))

        # Row 2: Agent Pos, Cell Info, Color Sensor
        pos_text = self.small_font.render(
            f"Agent: ({agent_pos[0]}, {agent_pos[1]})",
            True,
            self.colors["white"],
        )
        self.screen.blit(pos_text, (col1_x, panel_y + 45))

        current_cell = grid[agent_pos[0], agent_pos[1]]
        colour_name = (
            "None"
            if current_cell["colour"] == 0
            else "Red" if current_cell["colour"] == 1 else "Green"
        )
        cell_text = self.small_font.render(
            f"Color: {colour_name}", True, self.colors["white"]
        )
        self.screen.blit(cell_text, (col2_x, panel_y + 45))

        color_measurement = info.get("color_measurement")
        if color_measurement is not None:
            color_names = ["White", "Red", "Green"]
            color_text = self.small_font.render(
                f"Sensor: {color_names[color_measurement]}",
                True,
                self.colors["white"],
            )
            self.screen.blit(color_text, (col3_x, panel_y + 45))

        # Row 3: Ghost Pos, Items, Est. Pos
        ghost_text = self.small_font.render(
            f"Ghost: ({ghost_pos[0]}, {ghost_pos[1]})",
            True,
            self.colors["cyan"],
        )
        self.screen.blit(ghost_text, (col1_x, panel_y + 70))

        items_str = (
            ", ".join(current_cell["items"]) if current_cell["items"] else "None"
        )
        # Truncate items if too long
        if len(items_str) > 15:
            items_str = items_str[:12] + "..."
        items_text = self.small_font.render(
            f"Items: {items_str}", True, self.colors["white"]
        )
        self.screen.blit(items_text, (col2_x, panel_y + 70))

        est_pos_data = info.get("estimated_pos", {})
        est_pos = est_pos_data.get("cell_pos") if isinstance(est_pos_data, dict) else None
        if est_pos is not None:
            est_text = self.small_font.render(
                f"Est. Pos: ({est_pos[0]}, {est_pos[1]})",
                True,
                self.colors["orange"],
            )
            self.screen.blit(est_text, (col3_x, panel_y + 70))

        # Row 4: Distance, Goal
        distance = abs(agent_pos[0] - ghost_pos[0]) + abs(agent_pos[1] - ghost_pos[1])
        dist_text = self.small_font.render(
            f"Dist: {distance}", True, self.colors["yellow"]
        )
        self.screen.blit(dist_text, (col1_x, panel_y + 95))

        goal_text = self.small_font.render(
            f"Goal: {'Yes' if current_cell['is_goal'] else 'No'}",
            True,
            self.colors["yellow"] if current_cell["is_goal"] else self.colors["white"],
        )
        self.screen.blit(goal_text, (col2_x, panel_y + 95))

        # Row 5 & 6: Actions (Occupies bottom)
        intended_action = self.small_font.render(
            f"Intended: {info.get('intended_action', '')}",
            True,
            self.colors["white"],
        )
        self.screen.blit(intended_action, (col1_x, panel_y + 125))

        actual_action_str = info.get('actual_action', '')
        actual_action = self.small_font.render(
            f"Actual: {actual_action_str}",
            True,
            self.colors["white"],
        )
        self.screen.blit(actual_action, (col1_x, panel_y + 150))

"""

final_lines = lines[:start_line] + [new_method] + lines[end_line:]
with open(file_path, "w") as f:
    f.writelines(final_lines)
