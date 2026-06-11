from PIL import Image, ImageFilter, ImageEnhance

class ImageRes():
    def __init__(self, framein, frameout, downscale, upscale):
        self.framein_path = framein
        self.frameout_path = frameout
        self.downscale = downscale
        self.upscale = upscale

    def resdown(self):
        image = Image.open(self.framein_path)
        width, height = image.size
        new_width = int(width / self.downscale)
        new_height = int(height / self.downscale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        image.save(self.framein_path)


image_engine = ImageRes("VideoTracking/FrameIn/right.jpeg", "VideoTracking/FrameIn/right.jpeg", 1.5, 6)
image_engine.resdown()