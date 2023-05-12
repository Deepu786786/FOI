import gi
gi.require_version('Gst','1.0')
gi.require_version('GObject','2.0')
from gi.repository import Gst, GObject

#Initialize GStreamer
Gst.init(None)

#Create a pipeline
pipeline = Gst.Pipeline()

#Create pipeline elements
source = Gst.ElementFactory.make('udpsrc','source')
rtphdepay = Gst.ElementFactory.make('rtph264depay', 'h264_extractor')
h264parse = Gst.ElementFactory.make('h264parse', 'h264_parser')
queue = Gst.ElementFactory.make('queue', 'queue')
omxh264dec = Gst.ElementFactory.make('omxh264dec', 'h264_gpudecoder')
videoconvert = Gst.ElementFactory.make('nvvidconv','video_convert')
videosink = Gst.ElementFactory.make('xvimagesink', 'video_sink')


if not pipeline or not source or not rtphdepay or not h264parse or not queue or not omxh264dec or not videoconvert or not videosink:
	print('Failed to create elements')
	exit(1)


#Add elements to the pipeline
pipeline.add(source)
pipeline.add(rtphdepay)
pipeline.add(h264parse)
pipeline.add(queue)
pipeline.add(omxh264dec)
pipeline.add(videoconvert)
pipeline.add(videosink)

#Link elements
source.link(rtphdepay)
rtphdepay.link(h264parse)
h264parse.link(queue)
queue.link(omxh264dec)
omxh264dec.link(videoconvert)
videoconvert.link(videosink)

#RTSP URL
cam_url = "rtsp://admin:Dd22864549*@10.13.1.61:554/cam/realmonitor?channel=1\&subtype=0" 
source.set_property('location', cam_url)
# udpsrc.set_property('buffer-size',26214400)


pipeline.set_state(Gst.State.PLAYING)

bus = pipeline.get_bus()
msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)


if msg:
	if msg.type == Gst.MessageType.ERROR:
		error, debug_info = msg.parse_error()
		print(f"Error received from element {msg.src.get_name()}: {error.message}")
		print(f"Debugging information: {debug_info if debug_info else 'None'}")
	elif msg.type == Gst.MessageType.EOS:
		print("End-Of-Stream reached.")

pipeline.set_state(Gst.State.NULL)