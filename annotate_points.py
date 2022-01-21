import cv2
import cv2 as cv
import time
import os
import numpy as np 
import csv
import string


class Transform_Labeler():
    def __init__(self,directory, classes = ["No Pole", "Pole"],ds = 1):
        self.frame = 0
        self.buff_count = 0
        self.ds = ds
        
        self.directory = directory
        
        self.cap = cv2.VideoCapture(directory)
        ret,frame = self.cap.read()
        if self.ds != 1:
            frame = cv2.resize(frame,(frame.shape[1]//self.ds,frame.shape[0]//self.ds))
        self.frames = [frame]
        
        self.next_lane_id = [0,0]
        
        self.axes = []
        
        #self.load_annotations()
        
        self.cur_image = self.frames[0]
        
        self.start_point = None # used to store click temporarily
        self.clicked = False 
        self.new = None # used to store a new box to be plotted temporarily
        self.cont = True
        self.define_direction = False
        
        # classes
        self.cur_class = 0
        self.n_classes = len(classes)
        self.class_names = classes
        self.colors = (np.random.rand(2,3))*255
        self.colors[0] = np.array([0,0,255])
        self.colors[1] = np.array([255,0,0])

        self.plot_axes()
        self.changed = False
        self.y_lookup = {
                         0:0,
                         -1:-13.58,
                         -2:-31.58,
                         1:13.58,
                         -0.9:-13.58+.75,
                         1.1:13.58-.75,
                         -1.9:-31.58+1.333
                         }
    def get_x(self,x):
        x *= 12.75
        if self.class_names[self.cur_class] == "Pole":
            x -= 0.75
        return x
    
    def get_z(self,z):
        return z * 42/12.0 # yellow line top is at 42 inches
        

    # def load_annotations(self):
    #     try:
    #         self.cur_frame_boxes = []
    #         name = "annotations/new/{}.csv".format(self.frames[self.frame-1].split("/")[-1].split(".")[0])
    #         with open(name,"r") as f:
    #             read = csv.reader(f)
    #             for row in read:
    #                 if len(row) == 5:
    #                     row = [int(float(item)) for item in row]
    #                 elif len(row) > 5:
    #                     row = [int(float(item)) for item in row[:5]] + [float(item) for item in row[5:]]
    #                 self.cur_frame_boxes.append(np.array(row))
                    
    #     except FileNotFoundError:
    #         self.cur_frame_boxes = []
        
    def plot_axes(self):
        self.cur_image = self.frames[self.frame-self.buff_count].copy()

        last_source_idx = 0
        for box in self.axes:
                self.cur_image = cv2.circle(self.cur_image,(int(box[0]),int(box[1])),3,self.colors[self.cur_class],1)
                self.cur_image = cv2.circle(self.cur_image,(int(box[0]),int(box[1])),1,self.colors[self.cur_class],-1)

                
        for box in self.axes[-3:]:
            label = "({}ft,{}ft,{}ft)".format(box[2],box[3],box[4])
            font = cv2.FONT_HERSHEY_SIMPLEX
            self.cur_image = cv2.putText(self.cur_image,label,((int(box[0]),int(box[1]))),font,0.5,(255,255,255),1)
                
    def toggle_class(self):
        self.cur_class = (self.cur_class + 1) % self.n_classes
        self.next_lane_id = [0,0]
        print("Active Class: {}".format(self.class_names[self.cur_class]))
        
    def on_mouse(self,event, x, y, flags, params):
       if event == cv.EVENT_LBUTTONDOWN:
            # prompt for id number 
            # _,_ 
            #       - first is x coord
            # second is y coord
            try:
                print("Enter point id: _,_")
                lane_id = self.keyboard_input() # should be 
                lane_id = lane_id.split(",")
                lane_id = [int(item) for item in lane_id]
                self.next_lane_id = lane_id
                
                space_x = self.get_x(lane_id[0])
                y_str = lane_id[1] 
                
                try:
                    space_z = self.get_z(lane_id[2])
                except:
                    space_z = 0
                    
                if self.class_names[self.cur_class] == "Pole": y_str += 0.1
                space_y = self.y_lookup[y_str]
                
                
                # store
                box = np.array([x,y,space_x,space_y,space_z])
                self.axes.append(box)
                self.new = box
                print(box)
            except:
                print("Error,no point saved")
            
           
            
            
              
    def keyboard_input(self):
        keys = ""
        letters = string.ascii_lowercase + string.digits + string.punctuation
        while True:
            key = cv2.waitKey(1)
            for letter in letters:
                if key == ord(letter):
                    keys = keys + letter
            if key == ord("\n") or key == ord("\r"):
                break
        return keys           
            
    def quit(self):
        
        cv2.destroyAllWindows()
        self.cont = False
        self.save()
        
    def load(self):
        #try:    
            im_pts = []
            lmcs_pts = []
            
            
            name = "tform/" + self.directory.split("/")[-1].split("_")[2] + "_im_lmcs_transform_points.csv"
            with open(name,"r") as f:
                lines = f.readlines()
                
                for line in lines[1:-4]:
                    line = line.rstrip("\n").split(",")
                    im_pts.append ([float(line[0]),float(line[1])])
                    lmcs_pts.append([int(line[2]),int(line[3])])

            # plot computed points
            self.cur_image = self.frames[self.frame-1].copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            for idx in range(len(im_pts)):
                pt = im_pts[idx]
                label = str(lmcs_pts[idx])
                self.cur_image = cv2.circle(self.cur_image,(int(pt[0]),int(pt[1])),2,(0,255,255),-1)
                self.cur_image = cv2.circle(self.cur_image,(int(pt[0]),int(pt[1])),5,(0,255,255),1)
                self.cur_image = cv2.putText(self.cur_image,label,((int(pt[0]),int(pt[1]))),font,0.5,(255,255,255),1)
            cv2.imshow("computed_pts",self.cur_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            
            # compute transform
            H,mask = cv2.findHomography(im_pts,lmcs_pts)
            H_inv,mask_inv = cv2.findHomography(lmcs_pts,im_pts)
            
            test_points = []
            for x in range(-500,2500,20):
                for y in range(-240,240,12):
                    test_points.append(np.array([x,y]))
            test_points = np.stack(test_points)
            
            
            test_points_tf = self.transform_pt_array(test_points,H_inv)
            for idx,pt in enumerate(test_points_tf):
                self.cur_image = cv2.circle(self.cur_image,(int(pt[0]),int(pt[1])),1,(255,255,255),-1)
                #self.cur_image = cv2.putText(self.cur_image,str(test_points[idx]),((int(pt[0]),int(pt[1]))),font,0.25,(255,255,255),1)
    
            cv2.imshow("grid_pts",self.cur_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()                
            
        # except:
        #     pass
        
    def next(self):
        self.new = True

        if self.buff_count > 0:
            self.buff_count -=1
        else:
            ret,frame = self.cap.read()
            if ret:
                self.frames.append(frame)
                self.frame += 1

    def prev(self):
        self.buff_count += 1
        self.new = True
        
    def save(self,z_axis = False):
        
        im_pts = []
        for point in self.axes:
            im_pts.append(point[0:2])
        im_pts = np.stack(im_pts)
        
        lmcs_pts = []
        for point in self.axes:
            lmcs_pts.append(point[2:4])
        lmcs_pts = np.stack(lmcs_pts)
        

        
        if not z_axis:
            # compute transform
            H,mask = cv2.findHomography(im_pts,lmcs_pts)
            H_inv,mask_inv = cv2.findHomography(lmcs_pts,im_pts)
            
            test_points = []
            for x in range(10,300,10):
                for y in range(-36,36,2):
                    test_points.append(np.array([x,y]))
            test_points = np.stack(test_points)
            
            
            test_points_tf = self.transform_pt_array(test_points,H_inv)
            for idx,pt in enumerate(test_points_tf):
                self.cur_image = cv2.circle(self.cur_image,(int(pt[0]),int(pt[1])),1,(255,255,255),-1)
                #self.cur_image = cv2.putText(self.cur_image,str(test_points[idx]),((int(pt[0]),int(pt[1]))),font,0.25,(255,255,255),1)
    
            cv2.imshow("grid_pts",self.cur_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        
        save_points = self.axes
        
        name = "tform/" + self.directory.split("/")[-1].split("_")[1] + "_im_lmcs_transform_points.csv"
        
        if z_axis:
            name = "tform/z_axis_definition_points.csv"
        
        print(name)
        with open(name,"w") as f:
            writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["im x", "im y", "roadway x", "roadway y"])
            writer.writerows(save_points)
            
        print("Saved transform points as file {}".format(name))
        
        name = name.split(".csv")[0] + ".png"
        cv2.imwrite(name,self.cur_image)
    
    def transform_pt_array(self,point_array,M):
        """
        Applies 3 x 3  image transformation matrix M to each point stored in the point array
        """
        
        original_shape = point_array.shape
        
        num_points = int(np.size(point_array,0)*np.size(point_array,1)/2)
        # resize array into N x 2 array
        reshaped = point_array.reshape((num_points,2))   
        
        # add third row
        ones = np.ones([num_points,1])
        points3d = np.concatenate((reshaped,ones),1)
        
        # transform points
        tf_points3d = np.transpose(np.matmul(M,np.transpose(points3d)))
        
        # condense to two-dimensional coordinates
        tf_points = np.zeros([num_points,2])
        tf_points[:,0] = tf_points3d[:,0]/tf_points3d[:,2]
        tf_points[:,1] = tf_points3d[:,1]/tf_points3d[:,2]
        
        tf_point_array = tf_points.reshape(original_shape)
        
        return tf_point_array
        
    def undo(self):
        self.clicked = False
        self.define_direction = False
        self.define_magnitude = False
        
        self.axes = self.axes[:-1]
        self.cur_image = self.frames[self.frame-1].copy()
        self.plot_axes()
        
        
    def run(self):  
        cv2.namedWindow("window")
        cv.setMouseCallback("window", self.on_mouse, 0)
           
        while(self.cont): # one frame
        
           if self.new is not None:
               self.plot_axes()
                    
           self.new = None
               
           cv2.imshow("window", self.cur_image)
           title = "{} toggle class (1), switch frame (8-9), clear all (c), undo(u),   quit (q), switch frame (8-9)".format(self.class_names[self.cur_class])
           cv2.setWindowTitle("window",str(title))
           
           key = cv2.waitKey(1)
           if key == ord('9'):
                for i in range(10):
                    self.next()
           if key == ord('8'):
                self.prev()        
        
           elif key == ord('c'):
                self.clear()
           elif key == ord("q"):
                self.quit()
           elif key == ord("1"):
                self.toggle_class()
           elif key == ord("u"):
               self.undo()
           elif key == ord("d"):
               self.remove()
           elif key == ord("l"):
               self.load()
               
           elif key == ord("z"):
               self.save(z_axis = True)
               break
                      

#tfl  = Transform_Labeler("/home/worklab/Data/cv/video/5_min_18_cam_October_2020/ingest_session_00005/recording/record_p2c4_00000.mp4",ds = 2)
tfl = Transform_Labeler("/home/worklab/Desktop/RAV_4_Calib0.MP4")
tfl.run()

'''
The calculated homography can be used to warp
the source image to destination. Size is the
size (width,height) of im_dst
'''
#im_dst = cv2.warpPerspective(im_src, h, size)
