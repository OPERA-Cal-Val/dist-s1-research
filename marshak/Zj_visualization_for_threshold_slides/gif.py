from wand.image import Image


def pngs_to_gif(png_files, output_gif, frame_delay=100):
    with Image() as gif:
        for f in png_files:
            with Image(filename=f) as frame:
                frame.delay = frame_delay
                gif.sequence.append(frame)
        for frame in gif.sequence:
            frame.dispose = "background"
        gif.type = "optimize"
        gif.save(filename=output_gif)
