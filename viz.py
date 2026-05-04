"""Generate animated SVG visualization of flow matching sampling."""

import drawsvg as draw
import jax.numpy as jnp
from jaxtyping import Array


def create_animation(dataset_path: str, points: Array, output_path: str) -> None:
    """Create animated SVG showing sampled trajectories over dataset background.

    Args:
        dataset_path: Path to the .npy dataset file
        points: Sampled points with shape (batch_size, n_timestamps, 2)
        output_path: Path to save the animated SVG
    """
    dataset = jnp.load(dataset_path)
    dataset_flat = dataset.reshape(-1, 2)

    points_np = points
    if hasattr(points, "device_buffer"):
        points_np = jnp.array(points)

    batch_size, n_timestamps, _ = points.shape

    all_points = jnp.concatenate([dataset_flat, points_np.reshape(-1, 2)])
    x_min, x_max = float(all_points[:, 0].min()), float(all_points[:, 0].max())
    y_min, y_max = float(all_points[:, 1].min()), float(all_points[:, 1].max())

    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * padding
    x_max += x_range * padding
    y_min -= y_range * padding
    y_max += y_range * padding

    canvas_size = 400
    scale_x = canvas_size / (x_max - x_min)
    scale_y = canvas_size / (y_max - y_min)
    scale = min(scale_x, scale_y)

    def to_svg_coords(x: float, y: float):
        """Convert data coordinates to SVG coordinates with centered origin."""
        sx = x * scale
        sy = y * scale
        return sx, sy

    d = draw.Drawing(canvas_size, canvas_size)

    transform = f"translate({canvas_size / 2},{canvas_size / 2}) scale(1,-1)"
    main_group = draw.Group()
    main_group.append(draw.Raw(f'<g transform="{transform}">'))

    for i in range(dataset_flat.shape[0]):
        x, y = to_svg_coords(float(dataset_flat[i, 0]), float(dataset_flat[i, 1]))
        main_group.append(draw.Circle(x, y, 1.5, fill="#cccccc", fill_opacity=0.3))

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
    ]

    for b_idx in range(batch_size):
        color = colors[b_idx % len(colors)]

        path_data = []
        for t_past in range(n_timestamps):
            x, y = to_svg_coords(
                float(points_np[b_idx, t_past, 0]),
                float(points_np[b_idx, t_past, 1]),
            )
            if t_past == 0:
                path_data.append(f"M {x:.2f} {y:.2f}")
            else:
                path_data.append(f"L {x:.2f} {y:.2f}")

        path_elem = draw.Path(
            d=path_data[0],
            fill="none",
            stroke=color,
            stroke_width=1.5,
            stroke_opacity=0.8,
        )
        anim_values = ";".join(path_data)
        anim = draw.Animate(
            attributeName="d",
            values=anim_values,
            dur=f"{n_timestamps}s",
            repeatCount="indefinite",
        )
        path_elem.append_anim(anim)
        main_group.append(path_elem)

        circle = draw.Circle(0, 0, 3, fill=color, fill_opacity=0.9)
        cx_values = []
        cy_values = []
        for t_idx in range(n_timestamps):
            x, y = to_svg_coords(
                float(points_np[b_idx, t_idx, 0]),
                float(points_np[b_idx, t_idx, 1]),
            )
            cx_values.append(f"{x:.2f}")
            cy_values.append(f"{y:.2f}")
        anim_cx = draw.Animate(
            attributeName="cx",
            values=";".join(cx_values),
            dur=f"{n_timestamps}s",
            repeatCount="indefinite",
        )
        anim_cy = draw.Animate(
            attributeName="cy",
            values=";".join(cy_values),
            dur=f"{n_timestamps}s",
            repeatCount="indefinite",
        )
        circle.append_anim(anim_cx)
        circle.append_anim(anim_cy)
        main_group.append(circle)

    main_group.append(draw.Raw("</g>"))
    d.append(main_group)

    d.save_svg(output_path)
