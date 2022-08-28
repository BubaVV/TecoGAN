import ffmpeg
import logging
from sys import argv
import tensorflow as tf

from lib.upscaler import Upscaler, get_video_size, read_frame, write_frame, SCALE_Y, SCALE_X

if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    tf.get_logger().setLevel("ERROR")

    in_filename = argv[1]
    out_filename = argv[2]
    width, height = get_video_size(in_filename)
    upscaler = Upscaler(height, width)  # Order is correct
    process1 = (
        ffmpeg.input(in_filename)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run_async(pipe_stdout=True)
    )
    process2 = ffmpeg.input("pipe:",
                            format="rawvideo",
                            pix_fmt="rgb24",
                            s=f"{width * SCALE_X}x{height * SCALE_Y}")\
        .output("tmp.mp4", format="mp4")\
        .overwrite_output()\
        .run_async(pipe_stdin=True)

    while True:
        in_frame = read_frame(process1, width, height)
        if in_frame is None:
            logger.info("End of input stream")
            break

        logger.debug("Processing frame")
        out_frame = upscaler.process_frame(in_frame)
        write_frame(process2, out_frame)

    logger.info("Waiting for ffmpeg process1")
    process1.wait()

    logger.info("Waiting for ffmpeg process2")
    process2.stdin.close()
    process2.wait()

    upscaler.session.close()

    logger.info("Copying audio track")
    output = (
        ffmpeg.input("tmp.mp4")
        .output(ffmpeg.input(in_filename).audio, out_filename, vcodec="copy")
        .overwrite_output()
        .run()
    )

    logger.info("Done")