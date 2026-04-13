import sys

file_path = "src/custom_grid_env/colab_gui.py"
with open(file_path, "r") as f:
    content = f.read()

search_text = """                    # Plotting the contour plot
                    im = ax2.contourf(Y_grid, X_grid, dens, levels=20, cmap="viridis")
                    ax2.set_aspect("equal")
                    ax2.set_xlim(0, cols)
                    ax2.set_ylim(
                        rows, 0
                    )  # Inverted for grid coordinates (row 0 at top)

                    ax2.set_title("Estimated Probability Distribution (KDE)")
                    ax2.set_xlabel("Column")
                    ax2.set_ylabel("Row")
                    fig.colorbar(im, ax2, label="Probability Density")

                    # Draw estimated position as a small filled circle
                    est_pos = self.interface.pf.get_estimated_position()
                    float_pos = est_pos["float_pos"]  # [row, col]
                    ax2.scatter(
                        float_pos[1],
                        float_pos[0],
                        color="red",
                        s=100,
                        edgecolors="white",
                        label="Estimated Position",
                    )"""

replace_text = """                    # Plotting the contour plot
                    im = ax2.contourf(Y_grid, X_grid, dens, levels=20, cmap="viridis")
                    # Add contour lines for better visualization of the "elevation"
                    ax2.contour(
                        Y_grid, X_grid, dens, levels=10, colors="white", alpha=0.3
                    )
                    ax2.set_aspect("equal")
                    ax2.set_xlim(0, cols)
                    ax2.set_ylim(
                        rows, 0
                    )  # Inverted for grid coordinates (row 0 at top)

                    # Set explicit ticks for rows and columns
                    ax2.set_xticks(np.arange(0, cols))
                    ax2.set_yticks(np.arange(0, rows))

                    ax2.set_title("Estimated Probability Distribution (KDE)")
                    ax2.set_xlabel("Column")
                    ax2.set_ylabel("Row")
                    ax2.grid(True, linestyle="--", alpha=0.5)
                    fig.colorbar(im, ax2, label="Probability Density")

                    # Draw estimated position as a small filled circle
                    est_pos = self.interface.pf.get_estimated_position()
                    float_pos = est_pos["float_pos"]  # [row, col]
                    ax2.scatter(
                        float_pos[1],
                        float_pos[0],
                        color="red",
                        marker="X",
                        s=150,
                        edgecolors="white",
                        linewidths=2,
                        label="Estimated Position",
                        zorder=5,
                    )"""

new_content = content.replace(search_text, replace_text)
with open(file_path, "w") as f:
    f.write(new_content)
