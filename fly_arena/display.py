"""Stream a compact overview, physical body, eye inputs and live measurements."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def compose_frame(arena, eyes, mode, sim_seconds, total_spikes, command, wall_seconds):
    canvas = Image.new("RGB", (1024, 640), "#101c23")
    overview = Image.fromarray(arena.render()).resize((640, 480))
    body = Image.fromarray(arena.render(closeup=True)).resize((384, 288))
    canvas.paste(overview, (0, 68))
    canvas.paste(body, (640, 68))
    for i in range(2):
        canvas.paste(Image.fromarray(eyes[i]).resize((184, 184)), (648 + i * 184, 380))
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=24)
    font = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=14)
    draw.text((20, 12), "FLY ARENA / MaleCNS + MuJoCo", font=title_font, fill="#e1eee8")
    label = "SCRIPTED BODY TEST / NO CONNECTOME" if mode == "body-demo" else "EXPERIMENTAL LIF + ENGINEERED GAIT"
    draw.text((20, 42), label, font=small, fill="#e7bc72")
    draw.text((660, 357), "LEFT EYE                RIGHT EYE", font=small, fill="#94b4b9")
    draw.text((20, 558), f"time {sim_seconds:.2f} s   |   spikes / 10 ms {total_spikes:,}", font=font, fill="#e1eee8")
    draw.text((20, 585), f"gait L {command[0]:.2f}   R {command[1]:.2f}   |   wall {wall_seconds:.1f} s", font=font, fill="#94b4b9")
    draw.text((20, 613), "40 x 40 mm arena   /   6 legs   /   CPU simulation   /   no learned policy", font=small, fill="#94b4b9")
    return np.asarray(canvas)


def compose_ethology_frame(sim, *, resumed=False):
    canvas = Image.new("RGB", (1024, 768), "#101c23")
    canvas.paste(Image.fromarray(sim.arena.render()).resize((640, 480)), (0, 76))
    closeup = sim.arena.render(closeup=True)
    canvas.paste(Image.fromarray(closeup).resize((384, 288)), (640, 76))
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.load_default(size=25)
    font = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=13)
    draw.text((20, 12), "FLY ARENA / FEED - GROOM - REST", font=title, fill="#e1eee8")
    draw.text((20, 47), sim.label + (" / RESUMED SEGMENT" if resumed else ""), font=small, fill="#e7bc72")
    state, clocks = sim.organism.state, sim.organism.clocks
    bars = [("Energy", state.energy / sim.organism.config.energy_capacity, f"{state.energy:.1f} energy_u", "#e6af5b"),
            ("Gut", state.gut_amount / sim.organism.config.gut_capacity, f"{state.gut_amount:.2f} food_u", "#70bb95"),
            ("Sleep need", state.sleep_pressure, f"{state.sleep_pressure:.2f}", "#839ad4")]
    for i, (label, fraction, value, color) in enumerate(bars):
        y = 390 + i * 54
        draw.text((660, y), f"{label}: {value}", font=font, fill="#e1eee8")
        draw.rectangle((660, y + 26, 994, y + 36), fill="#2c3b42")
        draw.rectangle((660, y + 26, 660 + 334 * np.clip(fraction, 0, 1), y + 36), fill=color)
    decision = sim.last_decision
    draw.text((20, 578), f"{decision.action if decision else 'IDLE'}  /  {sim.arena.motor.primitive_phase}", font=title, fill="#e1eee8")
    draw.text((20, 614), f"physics {clocks.physics_time_s:.2f}s   neural {clocks.neural_time_s:.2f}s   life {clocks.life_time_s:.1f}s", font=font, fill="#a9c2c3")
    draw.text((20, 643), f"life scale x{sim.organism.config.life_time_scale:g}   wall {clocks.wall_time_s:.1f}s   upright {sim.arena.upright:.3f}", font=font, fill="#a9c2c3")
    draw.text((20, 674), "Dust " + "  ".join(f"{k}: {v:.2f}" for k, v in state.dust_by_region.items()), font=small, fill="#ccb598")
    draw.text((20, 702), f"Intake {state.ingested_total:.3f} food_u   removed dust {sim.environment.removed_dust:.3f}   "
              f"food remaining {sum(p.amount for p in sim.environment.food):.2f}", font=font, fill="#a9c2c3")
    if hasattr(sim.navigator, "phase"):
        footer = f"ENGINEERED SEARCH: {sim.navigator.phase} / local odor and airflow only / arbitrary resource units"
    else:
        footer = "Explicit organism model / physical contacts / arbitrary resource units / uncalibrated sleep"
    draw.text((20, 740), footer, font=small, fill="#80989a")
    return np.asarray(canvas)
