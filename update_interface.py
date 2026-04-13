import sys

file_path = "src/custom_grid_env/interface.py"
with open(file_path, "r") as f:
    content = f.read()

search_text = """        # Log all particle positions to the logger (DEBUG level -> file)
        logger.debug(f"Particle positions: {self.pf.get_particles()}")

        # Get and log estimated position
        est_pos = self.pf.get_estimated_position()
        cell_pos = est_pos["cell_pos"]
        logger.info(
            f"Estimated Agent Position (row, col): ({cell_pos[0]}, {cell_pos[1]})"
        )
        info["estimated_pos"] = est_pos

        # Trigger CNN prediction
        current_cell = self.env.grid[self.env.agent_pos[0], self.env.agent_pos[1]]
        prediction_info = self.vision_sensor.predict(current_cell)
        if prediction_info:
            info["cnn_probs"] = prediction_info["probs"]
            info["cnn_prediction"] = prediction_info["prediction"]
        cnn_probs = info.get("cnn_probs")

        measurements = {
            "color_measurement": info.get("color_measurement"),
            "cnn_probs": cnn_probs,
        }
        cnn_class_names = self.vision_sensor.class_names
        self.pf.update(
            measurements,
            self.pf_sensor_mode,
            self.env.grid,
            cnn_class_names,
        )
        self.pf.resample()"""

replace_text = """        # Trigger CNN prediction
        current_cell = self.env.grid[self.env.agent_pos[0], self.env.agent_pos[1]]
        prediction_info = self.vision_sensor.predict(current_cell)
        if prediction_info:
            info["cnn_probs"] = prediction_info["probs"]
            info["cnn_prediction"] = prediction_info["prediction"]
        cnn_probs = info.get("cnn_probs")

        measurements = {
            "color_measurement": info.get("color_measurement"),
            "cnn_probs": cnn_probs,
        }
        cnn_class_names = self.vision_sensor.class_names
        self.pf.update(
            measurements,
            self.pf_sensor_mode,
            self.env.grid,
            cnn_class_names,
        )
        self.pf.resample()

        # Log all particle positions to the logger (DEBUG level -> file)
        logger.debug(f"Particle positions: {self.pf.get_particles()}")

        # Get and log estimated position AFTER update and resample
        est_pos = self.pf.get_estimated_position()
        cell_pos = est_pos["cell_pos"]
        logger.info(
            f"Estimated Agent Position (row, col): ({cell_pos[0]}, {cell_pos[1]})"
        )
        info["estimated_pos"] = est_pos"""

new_content = content.replace(search_text, replace_text)
with open(file_path, "w") as f:
    f.write(new_content)
