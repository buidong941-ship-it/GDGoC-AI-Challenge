from PIL import Image, ImageDraw, ImageFont


TOP_BAR_HEIGHT = 48


def _draw_text(draw, position, text, fill=(0, 0, 0), font=None):
    draw.text(position, text, fill=fill, font=font)


def blast_tiles(grid, bx, by, radius):
    tiles = {(bx, by)}
    for drow, dcol in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        for r in range(1, radius + 1):
            tr, tc = bx + drow * r, by + dcol * r
            if not (0 <= tr < len(grid) and 0 <= tc < len(grid[0])):
                break
            cell = int(grid[tr][tc])
            if cell == 1:
                break
            tiles.add((tr, tc))
            if cell == 2:
                break
    return tiles


def explosion_tiles_from_transition(prev_obs, curr_obs):
    if prev_obs is None:
        return set()

    prev_bombs = prev_obs.get("bombs", [])
    curr_bombs = curr_obs.get("bombs", [])
    curr_positions = {(int(b[0]), int(b[1])) for b in curr_bombs}
    prev_players = prev_obs["players"]
    prev_grid = prev_obs["map"]

    tiles = set()
    for b in prev_bombs:
        bx, by, timer, owner_id = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        exploded = timer <= 1 or (bx, by) not in curr_positions
        if not exploded:
            continue
        radius = 1
        if 0 <= owner_id < len(prev_players):
            radius = 1 + int(prev_players[owner_id][4])
        tiles.update(blast_tiles(prev_grid, bx, by, radius))
    return tiles


def render_match_frame(obs, prev_obs=None, cell_size=40, top_bar_height=TOP_BAR_HEIGHT, agent_metadata=None):
    width = len(obs["map"][0])
    height = len(obs["map"])

    RIGHT_PANEL_WIDTH = 340
    board_width = width * cell_size
    total_width = board_width + RIGHT_PANEL_WIDTH

    img = Image.new("RGBA", (total_width, height * cell_size + top_bar_height), (245, 245, 245, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    # Draw top bar fully across
    draw.rectangle([0, 0, total_width, top_bar_height], fill=(30, 30, 30))
    
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 12)
        font_status = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font_title = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_status = ImageFont.load_default()

    _draw_text(draw, (10, 14), f"Step {int(obs.get('_step', 0))}", fill=(245, 245, 245), font=font_title)

    board_top = top_bar_height
    draw.rectangle([0, board_top, board_width, board_top + height * cell_size], fill=(144, 238, 144))

    grid = obs["map"]
    for row in range(height):
        for col in range(width):
            cell = int(grid[row][col])
            rect = [
                col * cell_size,
                board_top + row * cell_size,
                (col + 1) * cell_size,
                board_top + (row + 1) * cell_size,
            ]
            # Background and grid borders
            draw.rectangle(rect, fill=(144, 238, 144), outline=(120, 200, 120), width=1)
            
            if cell == 1:
                draw.rectangle(rect, fill=(80, 80, 80), outline=(40, 40, 40), width=2)
            elif cell == 2:
                draw.rectangle(rect, fill=(139, 69, 19), outline=(101, 67, 33), width=2)
                draw.line((rect[0], rect[1], rect[2], rect[3]), fill=(101, 67, 33), width=2)
                draw.line((rect[2], rect[1], rect[0], rect[3]), fill=(101, 67, 33), width=2)
            elif cell == 3:
                draw.rectangle(rect, fill=(225, 225, 225))
                cx = rect[0] + cell_size // 2
                cy = rect[1] + cell_size // 2
                radius = cell_size // 4
                draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(255, 0, 0))
                _draw_text(draw, (cx - 5, cy - 8), "R", fill=(255, 255, 255), font=font_small)
            elif cell == 4:
                draw.rectangle(rect, fill=(225, 225, 225))
                cx = rect[0] + cell_size // 2
                cy = rect[1] + cell_size // 2
                radius = cell_size // 4
                draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(0, 0, 255))
                _draw_text(draw, (cx - 5, cy - 8), "C", fill=(255, 255, 255), font=font_small)

    explosion_tiles = explosion_tiles_from_transition(prev_obs, obs)
    for row, col in explosion_tiles:
        px = col * cell_size
        py = board_top + row * cell_size
        draw.rectangle([px, py, px + cell_size, py + cell_size], fill=(255, 140, 0, 90))
        draw.ellipse([px + 10, py + 10, px + cell_size - 10, py + cell_size - 10], fill=(255, 220, 120))

    for b in obs.get("bombs", []):
        bx, by, timer = int(b[0]), int(b[1]), int(b[2])
        pulse = 2 if (int(obs.get('_step', 0)) + timer) % 2 == 0 else 0
        cx = by * cell_size + cell_size // 2
        cy = board_top + bx * cell_size + cell_size // 2
        radius = max(7, cell_size // 4 + pulse)
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(20, 20, 20), outline=(0, 0, 0))
        _draw_text(draw, (cx - 5, cy - 6), str(timer), fill=(255, 255, 255), font=font_small)

    colors = [(220, 50, 50), (50, 50, 220), (30, 150, 30), (200, 140, 0)]
    for i, p in enumerate(obs.get("players", [])):
        if not p[2]:
            continue
        row, col = int(p[0]), int(p[1])
        bombs_left = int(p[3]) if len(p) > 3 else 0
        radius_bonus = int(p[4]) if len(p) > 4 else 0
        
        cx = col * cell_size + cell_size // 2
        cy = board_top + row * cell_size + cell_size // 2
        radius = cell_size // 3
        
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=colors[i % len(colors)])
        
        # Draw ID with normal font
        _draw_text(draw, (cx - 5, cy - 8), str(i), fill=(255, 255, 255), font=font_small)
        
        # Display B and R under the agent circle
        _draw_text(draw, (cx - 16, cy + 12), f"B:{bombs_left} R:{radius_bonus}", fill=(0, 0, 0), font=font_small)

    # --- Right-side agent panel ---
    panel_x0 = board_width
    panel_y0 = top_bar_height
    panel_x1 = total_width
    panel_y1 = panel_y0 + height * cell_size
    panel_bg = (52, 58, 64)
    draw.rectangle([panel_x0, panel_y0, panel_x1, panel_y1], fill=panel_bg)
    draw.line([panel_x0, 0, panel_x0, panel_y1], fill=(30, 30, 30), width=2)

    title_text = "Agents"
    draw.text((panel_x0 + 10, panel_y0 + 8), title_text, fill=(245, 245, 245), font=font_title)

    players = obs.get("players", [])
    n_players = len(players)
    meta_names = []
    meta_colors = None
    if agent_metadata:
        meta_names = agent_metadata.get("agent_names") or agent_metadata.get("team_ids") or []
        meta_colors = agent_metadata.get("colors")
    if not meta_names or len(meta_names) < n_players:
        meta_names = [f"Agent {i}" for i in range(n_players)]

    y = panel_y0 + 40
    line_h = 22
    for i in range(n_players):
        name = meta_names[i] if i < len(meta_names) and meta_names[i] else f"Agent {i}"
        p = players[i] if i < len(players) else [0, 0, 0, 0, 0]
        alive = int(p[2]) == 1
        bombs_left = int(p[3]) if len(p) > 3 else 0
        radius_bonus = int(p[4]) if len(p) > 4 else 0
        color = tuple(meta_colors[i]) if (meta_colors and i < len(meta_colors)) else colors[i % len(colors)]

        cx = panel_x0 + 14
        cy = y + 8
        draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=color)

        name_str = str(name)[:40]
        draw.text((panel_x0 + 28, y), name_str, fill=(240, 240, 240), font=font_status)
        y += line_h

        status = "Alive" if alive else "Dead"
        status_color = (120, 220, 140) if alive else (220, 100, 100)
        draw.text((panel_x0 + 10, y), status, fill=status_color, font=font_status)
        y += line_h

        stats = f"Bombs: {bombs_left}  |  +Radius: {radius_bonus}"
        draw.text((panel_x0 + 10, y), stats, fill=(200, 200, 200), font=font_status)
        y += line_h + 6

    return img.convert("RGB")