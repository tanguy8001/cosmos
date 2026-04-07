import imageio
import os

video_path = "/n/netscratch/hankyang_lab/Lab/tdieudonne/cosmos-policy/cosmos_policy/experiments/robot/robocasa/logs/rollout_data/PnPCabToCounter--2026_04_03-20_33_34/2026_04_03-20_33_34--episode=3--success=False--task=pick_the_lemon_from_the_cabinet_and_plac.mp4"
reader = imageio.get_reader(video_path)
width, height = reader.get_meta_data()['size']
print(f"Original: {width}x{height}")

# If 3 views concatenated HORIZONTALLY, each view is [width/3]x[height]
# Rightmost view starts at x = (2 * width / 3)
view_width = width // 3
crop_x = 2 * view_width

print(f"Crop command: ffmpeg -i INPUT -vf 'crop={view_width}:{height}:{crop_x}:0' -c:v libx264 -crf 18 OUTPUT.mp4")
reader.close()
