import os
import moviepy.video.io.ImageSequenceClip
import glob

def make_video(frame_dir, video_dir):
    fps = 100
    image_files = [os.path.join(frame_dir, str(img_index)+".png")
                for img_index in range(len(os.listdir(frame_dir)))]
    print(len(image_files))
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(video_dir + 'new_video.mp4')
    files = glob.glob(frame_dir+"*")
    for f in files:
        os.remove(f)