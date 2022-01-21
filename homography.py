import torch
import numpy as np
import cv2
import sys, os
import csv
import _pickle as pickle


def line_to_point(line,point):
    """
    Given a line defined by two points, finds the distance from that line to the third point
    line - (x0,y0,x1,y1) as floats
    point - (x,y) as floats
    Returns
    -------
    distance - float >= 0
    """
    
    numerator = np.abs((line[2]-line[0])*(line[1]-point[1]) - (line[3]-line[1])*(line[0]-point[0]))
    denominator = np.sqrt((line[2]-line[0])**2 +(line[3]-line[1])**2)
    
    return numerator / (denominator + 1e-08)

def find_vanishing_point(lines):
    """
    Finds best (L2 norm) vanishing point given a list of lines

    Parameters
    ----------
    lines : [(x0,y0,x1,y1), ...]

    Returns
    -------
    vp - (x,y)
    """
    
    # mx+b form
    #y0 = ax + c
    #y1 = bx + d
    
    line0 = lines[0]
    line1 = lines[1]
    a = (line0[3] - line0[1])/line0[2] - line0[0]
    b = (line1[3] - line1[1])/line1[2] - line1[0]
    c = line0[1] - a*line0[0]
    d = line1[1] - c*line1[0]
    
    # intersection
    px = (d-c)/(a-b)
    py = a*(d-c)/(a-b) + c
    best_dist = np.inf
    
    # using intersection as starting point, grid out a grid of 11 x 11 points with spacing g
    g = 1e+16
    n_pts = 31
    
    while g > 1:
        #print("Gridding at g = {}".format(g))

        # create grid centered around px,py with spacing g
        
        x_pts = np.arange(px-g*(n_pts//2),px+g*(n_pts//2),g)
        y_pts = np.arange(py-g*(n_pts//2),py+g*(n_pts//2),g)
        
        for x in x_pts:
            for y in y_pts:
                # for each point in grid, compute average distance to vanishing point
                dist = 0
                for line in lines:
                    dist += line_to_point(line,(x,y))**2
                   
                # keep best point in grid
                if dist < best_dist:
                    px = x 
                    py = y
                    best_dist = dist
                    #print("Best vp so far: ({},{}), with average distance {}".format(px,py,np.sqrt(dist/len(lines))))
    
                # regrid
        g = g / 10.0
            
    return [px,py]

class Homography():
    """
    Homography provides utiliites for converting between image,space, and state coordinates
    One homography object corresponds to a single space/state formulation but
    can have multiple camera/image correspondences
    """

    def __init__(self,f1 = None,f2 = None):
        """
        Initializes Homgrapher object. 
        
        f1 - arbitrary function that converts a [d,m,3] matrix of points in space 
             to a [d,m,s] matrix in state formulation
        f2 - arbitrary function that converts [d,m,s] matrix into [d,m,3] matrix in space
        
        where d is the number of objects
              m is the number of points per object
              s is the state size

        returns - nothing

        """
        
        # if f1 is not None:
        #     self.f1 = f1
        #     self.f2 = f2
        
        # else:
        #     self.f1 = self.i24_space_to_state
        #     self.f2 = self.i24_state_to_space
        
        # each correspondence is: name: {H,H_inv,P,corr_pts,space_pts,vps} 
        # where H and H inv are 3x34 planar homography matrices and P is a 3x4 projection matrix
        self.correspondence = {}
    
        # self.class_heights = {
        #         "sedan":4,
        #         "midsize":5,
        #         "van":6,
        #         "pickup":5,
        #         "semi":12,
        #         "truck (other)":12,
        #         "truck": 12,
        #         "motorcycle":4,
        #         "trailer":3,
        #         "other":5
        #         }
        
        
        # self.class_dims = {
        #         "sedan":[16,6,4],
        #         "midsize":[18,6.5,5],
        #         "van":[20,6,6.5],
        #         "pickup":[20,6,5],
        #         "semi":[55,9,12],
        #         "truck (other)":[25,9,12],
        #         "truck": [25,9,12],
        #         "motorcycle":[7,3,4],
        #         "trailer":[16,7,3],
        #         "other":[18,6.5,5]
        #     }
        
        # self.class_dict = { "sedan":0,
        #             "midsize":1,
        #             "van":2,
        #             "pickup":3,
        #             "semi":4,
        #             "truck (other)":5,
        #             "truck": 5,
        #             "motorcycle":6,
        #             "trailer":7,
        #             0:"sedan",
        #             1:"midsize",
        #             2:"van",
        #             3:"pickup",
        #             4:"semi",
        #             5:"truck (other)",
        #             6:"motorcycle",
        #             7:"trailer"
        #             }
        
        self.default_correspondence = None
    
    def add_camera(self,camera_name = "RAV4"):
        # load points
        corr_pts= []
        space_pts = []
        
        xy_path = "tform/{}/calib_xy.csv".format(camera_name)
        z_path = "tform/{}/calib_z.csv".format(camera_name)
        
        
        with open(xy_path,"r") as f:
            lines = f.readlines()
            
            for line in lines[1:-4]:
                line = line.rstrip("\n").split(",")
                corr_pts.append ([float(line[0]),float(line[1])])
                space_pts.append([float(line[2]),float(line[3])])
        
        # load vps        
        corr_points = []
        space_points = []
        with open(z_path,"r") as f:
            read = csv.reader(f)
            lines = f.readlines()
            
            for line in lines[1:-4]:
                line = line.rstrip("\n").split(",")
                corr_points.append ([float(line[0]),float(line[1])])
                space_points.append([float(line[2]),float(line[3]),float(line[4])])
        
        # find lines that have same x and y coordinates
        lines3 = []
        for i in range(len(corr_points)):
            for j in range(len(corr_points)):
                if space_points[i][0] == space_points[j][0] and space_points[i][1]==space_points[j][1] and i != j:
                    lines3.append([corr_points[i][0],corr_points[i][1],corr_points[j][0],corr_points[j][1]])
                    
        # get all axis labels for a particular axis orientation
        vp1 = None
        vp2 = None
        vp3 = find_vanishing_point(lines3)
        vps = [vp1,vp2,vp3]
        
        self.add_correspondence(corr_pts,space_pts,vps,name = camera_name)
        self.scale_z(corr_points,space_points,name = camera_name)
        self.default_correspondence = camera_name
        
    # def i24_space_to_state(self,points):
    #     """
    #     points - [d,8,3] array of x,y,z points for fbr,fbl,bbr,bbl,ftr,ftl,fbr,fbl
        
    #     returns - [d,6] array of points in state formulation
    #     """
    #     d = points.shape[0]
    #     new_pts = torch.zeros([d,6])
        
    #     # rear center bottom of vehicle is (x,y)
        
    #     # x is computed as average of two bottom rear points
    #     new_pts[:,0] = (points[:,2,0] + points[:,3,0]) / 2.0
        
    #     # y is computed as average 4 bottom point y values
    #     new_pts[:,1] = (points[:,0,1] + points[:,1,1] +points[:,2,1] + points[:,3,1]) / 4.0
        
    #     # l is computed as avg length between bottom front and bottom rear
    #     new_pts[:,2] = torch.abs ( ((points[:,0,0] + points[:,1,0]) - (points[:,2,0] + points[:,3,0]))/2.0 )
        
    #     # w is computed as avg length between botom left and bottom right
    #     new_pts[:,3] = torch.abs(  ((points[:,0,1] + points[:,2,1]) - (points[:,1,1] + points[:,3,1]))/2.0)

    #     # h is computed as avg length between all top and all bottom points
    #     new_pts[:,4] = torch.mean(torch.abs( (points[:,0:4,2] - points[:,4:8,2])),dim = 1)
        
    #     # direction is +1 if vehicle is traveling along direction of increasing x, otherwise -1
    #     new_pts[:,5] = torch.sign( ((points[:,0,0] + points[:,1,0]) - (points[:,2,0] + points[:,3,0]))/2.0 ) 
                
    #     return new_pts
        
    # def i24_state_to_space(self,points):
    #     d = points.shape[0]
    #     new_pts = torch.zeros([d,8,3])
        
    #     # assign x values
    #     new_pts[:,[0,1,4,5],0] = (points[:,0] + points[:,5]*points[:,2]).unsqueeze(1).repeat(1,4)
    #     new_pts[:,[2,3,6,7],0] = (points[:,0]).unsqueeze(1).repeat(1,4)
        
    #     # assign y values
    #     new_pts[:,[0,2,4,6],1] = (points[:,1] - points[:,5]*points[:,3]/2.0).unsqueeze(1).repeat(1,4)
    #     new_pts[:,[1,3,5,7],1] = (points[:,1] + points[:,5]*points[:,3]/2.0).unsqueeze(1).repeat(1,4)
    
    #     # assign z values
    #     new_pts[:,4:8,2] = -(points[:,4]).unsqueeze(1).repeat(1,4) 
    
    #     return new_pts
    
    
    # def space_to_state(self,points):
    #     """
    #     points - [d,m,3] matrix of points in 3-space
    #     """
    #     return self.f1(points)
    
    # def state_to_space(self,points):
    #     """
    #     points - [d,m,s] matrix of points in state formulation
    #     """
    #     return self.f2(points)
    

    def add_correspondence(self,corr_pts,space_pts,vps,name = None):
        """
        corr_pts  - 
        space_pts - 
        vps       -
        name      - str, preferably camera name e.g. p1c4
        """
        
        if name is None:
            name = self.default_correspondence
            
        corr_pts = np.stack(corr_pts)
        space_pts = np.stack(space_pts)
        cor = {}
        cor["vps"] = vps
        cor["corr_pts"] = corr_pts
        cor["space_pts"] = space_pts
        
        cor["H"],_     = cv2.findHomography(corr_pts,space_pts)
        cor["H_inv"],_ = cv2.findHomography(space_pts,corr_pts)
        
        
        # P is a [3,4] matrix 
        #  column 0 - vanishing point for space x-axis (axis 0) in image coordinates (im_x,im_y,im_scale_factor)
        #  column 1 - vanishing point for space y-axis (axis 1) in image coordinates (im_x,im_y,im_scale_factor)
        #  column 2 - vanishing point for space z-axis (axis 2) in image coordinates (im_x,im_y,im_scale_factor)
        #  column 3 - space origin in image coordinates (im_x,im_y,scale_factor)
        #  columns 0,1 and 3 are identical to the columns of H, 
        #  We simply insert the z-axis column (im_x,im_y,1) as the new column 2
        
        P = np.zeros([3,4])
        P[:,0] = cor["H_inv"][:,0]
        P[:,1] = cor["H_inv"][:,1]
        P[:,3] = cor["H_inv"][:,2]
        P[:,2] = np.array([vps[2][0],vps[2][1],1]) * 0.01
        cor["P"] = P
        
        self.correspondence[name] = cor
        
        if self.default_correspondence is None:
            self.default_correspondence = name
            
    
    
    def remove_correspondence(self,name):        
        try:
            del self.correspondences[name]
            print("Deleted correspondence for {}".format(name))
        except KeyError:
            print("Tried to delete correspondence {}, but this does not exist".format(name))
    
    
    def scale_z(self,image_points,space_points,name = None):
        """
        Adjusts z-axis scaling in P until space points overlap as closely as possible with image points
        """
        
        if name is None:
            name = self.default_correspondence
            
        im_pts = []
        for item in image_points:
            im_pts.append(torch.tensor(item))
        image_points = torch.stack(im_pts)
        
        space_pts = []
        for item in space_points:
            space_pts.append(torch.tensor(item))
        space_points = torch.stack(space_pts)
        
        
        im_est = self.space_to_im(space_points.unsqueeze(0)).squeeze(0)
        
        return
        # get average 
        
    
    def im_to_space(self,points, name = None,heights = None):
        """
        Converts points by means of ____________
        
        points - [d,m,2] array of points in image
        """
        if name is None:
            name = self.default_correspondence
        
        
        d = points.shape[0]
        
        # convert points into size [dm,3]
        points = points.reshape(-1,2).double()
        points = torch.cat((points,torch.ones([points.shape[0],1]).double()),1) # add 3rd row
        
        if heights is not None:
            
            if type(name) == list:
                H = torch.from_numpy(np.stack([self.correspondence[sub_n]["H"].transpose(1,0) for sub_n in name])) # note that must do transpose(1,0) because this is a numpy operation, not a torch operation ...
                H = H.unsqueeze(1).repeat(1,8,1,1).reshape(-1,3,3)
                points = points.unsqueeze(1)
                new_pts = torch.bmm(points,H)
                new_pts = new_pts.squeeze(1)
            else:
                H = torch.from_numpy(self.correspondence[name]["H"]).transpose(0,1)
                new_pts = torch.matmul(points,H)
            
            # divide each point 0th and 1st column by the 2nd column
            new_pts[:,0] = new_pts[:,0] / new_pts[:,2]
            new_pts[:,1] = new_pts[:,1] / new_pts[:,2]
            
            # drop scale factor column
            new_pts = new_pts[:,:2] 
            
            # reshape to [d,m,2]
            new_pts = new_pts.reshape(d,-1,2)
            
            # add third column for height
            new_pts = torch.cat((new_pts,torch.zeros([d,new_pts.shape[1],1]).double()),2)
            
            new_pts[:,[4,5,6,7],2] = heights.unsqueeze(1).repeat(1,4).double()
            
        else:
            print("No heights were input")
            return
        
        return new_pts
    
    
    def space_to_im(self,points,name = None):
        """
        Projects 3D space points into image/correspondence using P:
            new_pts = P x points T  ---> [dm,3] T = [3,4] x [4,dm]
        performed by flattening batch dimension d and object point dimension m together
        
        points - [d,m,3] array of points in 3-space
        """
        if name is None:
            name = self.default_correspondence
        
        d = points.shape[0]
        
        # convert points into size [dm,4]
        points = points.reshape(-1,3)
        points = torch.cat((points.double(),torch.ones([points.shape[0],1]).double()),1) # add 4th row
        
        
        # project into [dm,3]
        if type(name) == list:
                P = torch.from_numpy(np.stack([self.correspondence[sub_n]["P"] for sub_n in name]))
                P = P.unsqueeze(1).repeat(1,8,1,1).reshape(-1,3,4)
                points = points.unsqueeze(1).transpose(1,2)
                new_pts = torch.bmm(P,points).squeeze(2)
        else:
            points = torch.transpose(points,0,1).double()
            P = torch.from_numpy(self.correspondence[name]["P"]).double()
            new_pts = torch.matmul(P,points).transpose(0,1)
        
        # divide each point 0th and 1st column by the 2nd column
        new_pts[:,0] = new_pts[:,0] / new_pts[:,2]
        new_pts[:,1] = new_pts[:,1] / new_pts[:,2]
        
        # drop scale factor column
        new_pts = new_pts[:,:2] 
        
        # reshape to [d,m,2]
        new_pts = new_pts.reshape(d,-1,2)
        return new_pts
    
    
    # def state_to_im(self,points,name = None):
    #     """
    #     Calls state_to_space, then space_to_im
        
    #     points - [d,s] matrix of points in state formulation
    #     """
    #     if name is None:
    #         name = self.default_correspondence
        
    #     return self.space_to_im(self.state_to_space(points),name = name)
    
    
    # def im_to_state(self,points,name = None, heights = None):
    #     """
    #     Calls im_to_space, then space_to_state
        
    #     points - [d,m,2] array of points in image
    #     """
    #     if name is None:
    #         name = self.default_correspondence
        
    #     return self.space_to_state(self.im_to_space(points,heights = heights,name = name))
    
    # def guess_heights(self,classes):
    #     """
    #     classes - [d] vector of string class names
        
    #     returns - [d] vector of float object height guesses
    #     """
        
    #     heights = torch.zeros(len(classes))
        
    #     for i in range(len(classes)):
    #         try:
    #             heights[i] = self.class_heights[classes[i]]
    #         except KeyError:
    #             heights[i] = self.class_heights["other"]
            
    #     return heights
    
    # def height_from_template(self,template_boxes,template_space_heights,boxes):
    #     """
    #     Predicts space height of boxes in image space. Given a space height and 
    #     the corresponding image box (and thus image height), the relationship 
    #     between heights in different coordinate systems should be roughly estimable. 
    #     This strategy is used to guess the heights of the second set of boxes in
    #     image space according to : 
    #         template_im_heights:template_space_heights = new_im_heights:new_box heights
            
    #     template_boxes - [d,m,2,] array of points corresponding to d object boxes 
    #                      (typical usage would be to use boxes from previous frame
    #                      or apriori box predictions for current frame))
    #     template_space_heights - [d] array of corresponding object heights in space
    #     boxes - [d,m,2] array of points in image
        
    #     returns
        
    #     height - [d] array of object heights in space
    #     """
        
    #     # get rough heights of objects in image
    #     template_top = torch.mean(template_boxes[:,4:8,:],dim = 1)
    #     template_bottom = torch.mean(template_boxes[:,0:4,:],dim = 1)
    #     template_im_height = torch.sum(torch.sqrt(torch.pow((template_top - template_bottom),2)),dim = 1)
    #     template_ratio = template_im_height / template_space_heights
        
    #     box_top    = torch.mean(boxes[:,4:8,:],dim = 1)
    #     box_bottom = torch.mean(boxes[:,0:4,:],dim = 1)
    #     box_height = torch.sum(torch.sqrt(torch.pow((box_top - box_bottom),2)),dim = 1)


    #     height = box_height / template_ratio
    #     return height
    
    
    # def test_transformation(self,points,classes = None,name = None, im = None,heights = None, verbose = True):
    #     """
    #     Transform image -> space -> state -> space -> image and 
    #     outputs the average reprojection error in pixels for top and bottom box coords
        
    #     points - [d,8,2] array of pixel coordinates corresponding to object corners
    #              fbr,fbl,bbr,bbl,ftr,ftl,fbr,fbl
    #     name - str camera/correspondence name
    #     im- if a cv2-style image is given, will plot original and reprojected boxes        
    #     heights - [d] array of object heights, otherwise heights will be guessed
    #               based on class
    #     """
    #     if name is None:
    #         name = self.default_correspondence
        
        
    #     if heights is None:
    #         if classes is None:
    #             print("Must either specify heights or classes for boxes")
    #             return
    #         else:
    #             guess_heights = self.guess_heights(classes)
                
                
    #     else:
    #         guess_heights = heights
            
    #     state_pts = self.im_to_state(points,heights = guess_heights,name = name)
    #     im_pts_repro = self.state_to_im(state_pts,name = name)
    
    #     # calc error
    #     error = torch.abs(points - im_pts_repro)        
    #     bottom_error = torch.sqrt(torch.pow(error[:,:4,0],2) + torch.pow(error[:,:4,1],2)).mean()
    #     top_error = torch.sqrt(torch.pow(error[:,4:8,0],2) + torch.pow(error[:,4:8,1],2)).mean()
        
    #     if verbose:
    #         print("Average distance between reprojected points and original points:")
    #         print("-----------------------------")
    #         print("Top: {} pixels".format(top_error))
    #         print("Bottom: {} pixels".format(bottom_error))
        
    #     # if image, plot
    #     if im is not None:
    #         im = self.plot_boxes(im,points,color = (0,255,0))
    #         im = self.plot_boxes(im,im_pts_repro,color = (0,0,255))
        
    #         cv2.imshow("frame",im)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
        
    #     return top_error + bottom_error
        
        
    # def scale_Z(self,boxes,heights,name = None, granularity = 1e-06, max_scale = 10):
    #     """
    #     When a new correspondence is added, the 3rd column of P is off by a scale factor
    #     relative to the other columns. This function scales P optimally
    #     to minimize the reprojection errror of the given boxes with the given heights
        
    #     boxes - [d,8,2] array of image points corresponding to object bounding boxes
    #             d indexes objects
    #     heights - [d] array of object heights (in space coordinates e.g. feet)
    #     name - str - correspondence 
    #     granularity - float - controls the minimum step size for grid search 
    #     max_scale - float - roughly, a reasonable upper estimate for the space-unit change
    #             corresponding to one pixel in the Z direction
                
    #     returns - None (but alters P in self.correspondence)
    #     """
    #     if name is None:
    #         name = self.default_correspondence
        
    #     P_orig = self.correspondence[name]["P"].copy()
        
    #     upper_bound = max_scale
    #     lower_bound = granularity
        
    #     # create a grid of 10 evenly spaced entries between upper and lower bound
    #     C_grid = np.linspace(lower_bound,upper_bound,num = 10)
    #     step_size = C_grid[1] - C_grid[0]
    #     iteration = 1
        
    #     while step_size > granularity:
            
    #         best_error = np.inf
    #         best_C = None
    #         # for each value of P, get average reprojection error
    #         for C in C_grid:
                
    #             # scale P
    #             P = P_orig.copy()
    #             P[:,2] *= C
    #             self.correspondence[name]["P"] = P
                
    #             # test error
    #             error = self.test_transformation(boxes,name = name, heights = heights,verbose = False)
                
    #             # if this is the best so far, store it
    #             if error < best_error:
    #                 best_error = error
    #                 best_C = C
                    
            
    #         # define new upper, lower  with width 2*step_size centered on best value
    #         #print("On loop {}: best C so far: {} avg error {}".format(iteration,best_C,best_error))
    #         lower_bound = best_C - step_size
    #         upper_bound = best_C + step_size
    #         C_grid = np.linspace(lower_bound,upper_bound,num = 10)
    #         step_size = C_grid[1] - C_grid[0]

    #         #print("New C_grid: {}".format(C_grid.round(4)))
    #         iteration += 1
    #     print("Best Error: {}".format(best_error))
        
        

    def plot_boxes(self,im,boxes,color = (255,255,255),labels = None,thickness = 1):
        """
        As one might expect, plots 3D boxes on input image
        
        im - cv2 matrix-style image
        boxes - [d,8,2] array of image points where d indexes objects
        color - 3-tuple specifying box color to plot
        """
                
        DRAW = [[0,1,1,0,1,0,0,0], #bfl
                [0,0,0,1,0,1,0,0], #bfr
                [0,0,0,1,0,0,1,1], #bbl
                [0,0,0,0,0,0,1,1], #bbr
                [0,0,0,0,0,1,1,0], #tfl
                [0,0,0,0,0,0,0,1], #tfr
                [0,0,0,0,0,0,0,1], #tbl
                [0,0,0,0,0,0,0,0]] #tbr
        
        DRAW_BASE = [[0,1,1,1], #bfl
                     [0,0,1,1], #bfr
                     [0,0,0,1], #bbl
                     [0,0,0,0]] #bbr
        
        for idx, bbox_3d in enumerate(boxes):
            
            # TODO - check whether box mostly falls within frame
            
            for a in range(len(bbox_3d)):
                ab = bbox_3d[a]
                for b in range(a,len(bbox_3d)):
                    bb = bbox_3d[b]
                    if DRAW[a][b] == 1:
                        try:
                            im = cv2.line(im,(int(ab[0]),int(ab[1])),(int(bb[0]),int(bb[1])),color,thickness)
                        except:
                            pass
        
            if labels is not None:
                label = labels[idx]
                left  = bbox_3d[0,0]
                top   = bbox_3d[0,1]
                im    = cv2.putText(im,"{}".format(label),(int(left),int(top - 10)),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),3)
                im    = cv2.putText(im,"{}".format(label),(int(left),int(top - 10)),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
            
        return im
    
    def plot_points(self,im,points,colors = None,sizes = None):
        """
        Plots points on input image
        im - cv2-style matrix image
        points - n,3 array of points in 3D space
        color - n,3 array of colors for points or None
        sizes - n,1 iterable of point sizes for plotting
        """
        
        im_points = self.space_to_im(points.unsqueeze(0)).squeeze(0)
        
        for pidx,point in enumerate(im_points):
            if colors is not None:
                color = colors[pidx]
            else:
                color = (0,0,255)
            if sizes is not None:
                size = sizes[pidx]
            else:
                size = 2
            cv2.circle(im,(int(point[0]),int(point[1])),size,color,-1)
            
    def plot_axes(self,im,major_xtick = 25,minor_xtick = 5,major_ytick = 12,minor_ytick = 6):
        """
        Plots an axis grid on the image
        """
        
        # create dx2x3 grid of points
        
        y_width = 48
        x_width = 101
            
        
        lines = []
        for i in range(0,x_width,minor_xtick):
            lines.append(torch.tensor([[i,-y_width/2,0],[i,y_width/2,0]],dtype = float))
        for j in range(-y_width//2,y_width//2,minor_ytick):
            lines.append(torch.tensor([[0,j,0],[x_width,j,0]],dtype = float))
                         
        lines = torch.stack(lines)
        im_lines = self.space_to_im(lines)
        
        for line in im_lines:
            line = line.int()
            cv2.line(im,(line[0,0],line[0,1]),(line[1,0],line[1,1]),(100,100,100),1)

        lines = []
        for i in range(0,x_width,major_xtick):
            lines.append(torch.tensor([[i,-y_width/2,0],[i,y_width/2,0]],dtype = float))
        for j in range(-y_width//2,y_width//2,major_ytick):
            lines.append(torch.tensor([[0,j,0],[x_width,j,0]],dtype = float))
                         
        lines = torch.stack(lines)
        im_lines = self.space_to_im(lines)
        
        for line in im_lines:
            line = line.int()
            cv2.line(im,(line[0,0],line[0,1]),(line[1,0],line[1,1]),(255,255,255),2)
      
        
    # def plot_test_point(self,point,im_dir):
    #     """
    #     Plot a single point defined in 3D space, in each camera view in which
    #     that point is visible
        
    #     im_dir - (string) directory with an example image for each correspondence
    #     point  - *list of length 3) defining a point in 3D space
    #     """
    #     point = torch.tensor(point).unsqueeze(0).unsqueeze(0)

    #     # get image for each correspondence
    #     for corr in self.correspondence.keys():
    #         print(corr)
            
    #         # get a path to an image
    #         files = os.listdir(im_dir)
    #         for file in files:
    #             if corr in file and "csv" not in file:
    #                 im_path = os.path.join(im_dir,file)
    #                 break
            
    #         # get point coordinates in local image space
    #         im_point = self.space_to_im(point,name = corr).reshape(-1)
    #         center = (int(im_point[0]),int(im_point[1]))
            
    #         if center[0] > 0 and center[0] < 1920 and center[1] > 0 and center[1] < 1080:
            
    #             im = cv2.imread(im_path)
    #             im = cv2.circle(im,center,3,(0,0,255),-1)
    #             im = cv2.circle(im,center,10,(0,0,255),2)
    #             cv2.imshow(corr,im)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()

# def load_i24_csv(file):
#         """
#         Simple no-frills function to load data as currently formatted on the i24 project
#         labels - first row of string headers for data columns
#         data - dict of lists, one key per frame, one entry per frame object
#         """
#         short_name = file.split("/")[-1]
#         HEADERS = True
        
#         # parse first file
#         rows = []
#         with open(file,"r") as f:
#             read = csv.reader(f)
            
#             for row in read:
#                 rows.append(row)
                    
#         data = {}
#         HEADERS = True
#         for row_idx in range(len(rows)):
#             row = rows[row_idx]
            
#             # pass header lines through as-is
#             if HEADERS:
#                 headers = row
#                 if len(row) > 0 and row[0] == "Frame #":
#                     HEADERS = False
            
            
#             else:
                
#                 if len(row) == 0:
#                     continue
                
#                 frame_idx = int(row[0])
#                 if frame_idx not in data.keys():
#                     data[frame_idx] = [row]
#                 else:
#                     data[frame_idx].append(row)
                
        
#         return headers,data

# class Homography_Wrapper():
#     """
#     This class was added as a workaround for the following problem:  multiple
#     correspondences for a single camera defined locally such that one or the other 
#     correspondence is more accurate within certain regions of space. This class
#     implements the same basic, non-fitting functions as Homography and determines
#     based on input locations which correspondence to use for each object. 
#     Implemented functions:
#         im_to_space
#         im_to_state
#         state_to_space  (identical to Homgraphy, pass-through function)
#         state_to_im
#         space_to_im
#         space_to_state (identical to Homgraphy, pass-through function)
#         plot_boxes 
#         guess_heights (identical to Homography, pass-through function)
#         height_from_template (identical to Homography, pass-through function)
        
        
#         All input and output formulations and method functionality are the same
#         as for Homgraphy unless otherwise specified. Please refer to comments 
#         in Homgraphy class for usage and debugging.
#     """
#     def __init__(self,hg1 = None,hg2 = None):
#         """
#         hg1 - initialized Homgraphy object with all correspondences that will be 
#                 used already added
#         hg2 - initialized Homography object with the same set of correspondece names
#         """
#         self.hg1 = hg1
#         self.hg2 = hg2
        
#         if hg1 is None and hg2 is None:
#             self.hg1 = get_homographies(save_file = "EB_homography.cpkl",direction = "EB")
#             self.hg2 = get_homographies(save_file = "WB_homography.cpkl",direction = "WB")
        
#     ## Pass-through functions 
#     def guess_heights(self,classes):
#         return self.hg1.guess_heights(classes)
#     def state_to_space(self,points):
#         return self.hg1.state_to_space(points)
#     def space_to_state(self,points):
#         return self.hg1.space_to_state(points)
#     def height_from_template(self,template_boxes,template_space_heights,boxes):
#         return self.hg1.height_from_template(template_boxes,template_space_heights,boxes)

#     ## Wrapper functions
#     def im_to_space(self,points, name = None,heights = None):
#         boxes  = self.hg1.im_to_space(points,name = name, heights = heights)
#         boxes2 = self.hg2.im_to_space(points,name = name, heights = heights)

#         # get indices where to use boxes1 and where to use boxes2 based on centerline y
#         ind = torch.where(boxes[:,0,1] > 60)[0] 
#         boxes[ind,:,:] = boxes2[ind,:,:]
#         return boxes
    
#     def space_to_im(self,points,name = None):
#         boxes  = self.hg1.space_to_im(points,name = name)
#         boxes2 = self.hg2.space_to_im(points,name = name)
        
#         # get indices where to use boxes1 and where to use boxes2 based on centerline y
#         ind = torch.where(points[:,0,1] > 60)[0]
#         boxes[ind,:] = boxes2[ind,:]        
#         return boxes
    
#     def im_to_state(self,points,name = None, heights = None):
#         return self.space_to_state(self.im_to_space(points,name = name, heights = heights))

#     def state_to_im(self,points,name = None):
#         return self.space_to_im(self.state_to_space(points),name = name)
    
#     def plot_state_boxes(self,im,boxes,name = None, color = (255,255,255),secondary_color = None,labels = None,thickness = 1):
#         """
#         im - cv2 image
#         boxes - [d,s] boxes in state formulation
#         """
        
#         if secondary_color is None:
#             secondary_color = color
            
#         # plot objects that are best fit by hg2
#         ind = torch.where(boxes[:,1] > 60)[0] 
        
#         labels1 = None
#         if labels is not None:
#             labels1 = [labels[i] for i in ind]
#         boxes1 = boxes[ind,:]
#         if len(boxes1) > 0:
#             im_boxes1 = self.state_to_im(boxes1,name = name)
#             im = self.hg2.plot_boxes(im,im_boxes1,color = secondary_color,labels = labels1,thickness = thickness)
        
#         # plot objects that are best fit by hg1
#         ind = torch.where(boxes[:,1] < 60)[0]
#         labels2 =  None
#         if labels is not None:
#             labels2 = [labels[i] for i in ind]
            

#         boxes2 = boxes[ind,:]
#         if len(boxes2) > 0:
#             im_boxes2 = self.state_to_im(boxes2,name = name)
#             im = self.hg1.plot_boxes(im,im_boxes2,color = color,labels = labels2,thickness = thickness)

#         return im
    
    
        
        

# basic test code
if __name__ == "__main__":
    
   
    hg = Homography()
    hg.add_camera(camera_name = "RAV4")
    
    
    # ensure points can be transformed
    points = torch.tensor([[10,0,0],
                           [10,10,0]])
    
    im_points = hg.space_to_im(points.unsqueeze(0)).squeeze(0)
    
    
    path = "/media/worklab/7907-EE51/calib/2021_0806_130951_063A.MP4"
    cap = cv2.VideoCapture(path)
    ret,frame = cap.read()
    while ret:
        
        hg.plot_axes(frame)
        cv2.imshow("frame",frame)
        key = cv2.waitKey(1)
        
        if key == ord("q"):
            cv2.destroyAllWindows()
            cap.release()
            break
        
        ret,frame = cap.read()