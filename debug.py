from cr_bot.adb.scrcpy_multiclient import MultiClient

mc = MultiClient(target_width=720, max_fps=30)      # or MultiClient(serials=["R58M…", "10.0.0.5:5555"])
mc.start()

frame = mc.latest_frame()                            # newest BGR frame, never a stale one
print(mc.resolution, f"{mc.fps:.1f} fps", mc.status())

# Native drag: press the card, glide to the tile, release.
mc.control.drag((224, 1162), (300, 700))

mc.stop()
