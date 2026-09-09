## Limitations

The current implementation performs face detection independently on each video frame. Therefore, it does not explicitly maintain an identity for each character across frames.

Detection performance can also decrease when faces are:

* Very small in the frame
* Heavily occluded
* Viewed from extreme angles
* Affected by motion blur
* Poorly illuminated

For a production-level implementation, I would combine a stronger face detector with a multi-object tracking algorithm and optionally fine-tune the detector using frames representative of the target video.
