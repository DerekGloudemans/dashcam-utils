# dashcam-utils

Congratulations! You've scored a sneak peak at the hottest new data vis repository. Sorry the readme isn't quite up to snuff and the code is pretty basic at present. I'll give a tiny overview of how to use this repo

## Requirements
Pytorch, OpenCV

## Contents
- `annotate_points.py` - use this to define correspondence points in space and dashcam image
- `homography.py` - the meat and potatoes class of this repo. Homography class is a container for all of the correspondence information necessary to do perspective geometry calculations (e.g. if this point is at (x,y,z) in space, where should it fall in this camera image?)
- `/tform` - contains one correspondence folder for each camera correspondence you want to work with, so for now just `/RAV4`

## Usage of Homography

Import homography file
```
 import homography
```

Create a Homography object:
```
hg = homography.Homography()
```

Add correspondence to homography object (each object can contain more than one correspondence (i.e. front dashcam, rear cam) as it can be nice to work with points from both cameras at once). If you want to add a new camera, there needs to be a corresponding folder in `/tform` containing calib_xy.csv and calib_z.csv:
```
hg.add_camera(camera_name = "RAV4")
```

Now, converting points is easy! Homgraphy objects expect points in a `[d,n,3]` tensor. This three-dimensionality is nice if you want to plot, say, lines or boxes, because d can index line and n can index points within the line. If you simply want to plot flattened points, just add an empty dimension d:
```
# example points
points = torch.rand(100,2)                             # 100 random 2D points
points = torch.cat((points,torch.zeros(100,1),dim = 1) # here we add a z coordinate set to 0 for all points.
points = points.unsqueeze(0)                           # points now has dimension `[1,100,3]`

image_points = hg.space_to_im(points,name = "RAV4")    # if you don't specify name, the last camera you added will be used by default
image_points = image_points.squeeze(0)                 # reduce dimension from `[1,100,2]` to `[100,2]`
```


## Still to be implemented
- Z axis scaling - right now Z coordianates don't correspond to anything in particular. This is a trivial fix but I haven't done it yet
- Conversion of points from image to space - requires a method for parsing the ambiguity inherent in converting 2 dimensions into 3 dimensions
- Vehicle detection from images - using Monocular 3D detection and tracking
