#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 11:56:54 2025

@author: david
"""

import h5py
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import cv2
import shutil
import subprocess
from pathlib import Path
from shapely.geometry import LineString, Polygon


class posLoader:
    
    def __init__(self, filename):
        
        self.filename = filename
        with h5py.File(filename, "r") as f:
            if 'tracks' not in f:
                print("Dataset 'tracks' not found in file")
                return
            #self.data = f['tracks'][:]
            self.data = self._fill_missing_data(f['tracks'][:]) #Shape: (2, 2, 5, x) --> (mouse0/1, x/y, which_body_part, which_frame)
            
        self.minFramesStill = 10
        self.stillnessRange = 10
        
        #List of body parts tracked
        self.NOSE_INDEX = 0
        self.earL_INDEX = 1
        self.earR_INDEX = 2
        self.HB_INDEX = 3 #head base index
        self.TB_INDEX = 4 #tail base index
        
        self.vectorLength = 250
        
        #Frame Dimensions
        self.width = 1392
        self.height = 640
        
        # Define horizontal boundaries for zones
        self.levBoundary = self.width // 3
        self.magBoundary = 2 * self.width // 3
    
        # Define regions based on x-coordinate of headbase
        self.levRegion = (0, self.levBoundary)
        self.middleRegion = (self.levBoundary, self.magBoundary)
        self.magRegion = (self.magBoundary, self.width)
        
        
    def _fill_missing_data(self, Y, kind = "linear"):
        """Fills missing values independently along each (mouse, coord, part) trace over frames."""
    
        # Store initial shape: (mouse, coord, part, frame)
        initial_shape = Y.shape
        M, C, P, F = initial_shape
        
        # Move 'frame' to the first axis and flatten the rest → shape: (frame, M * C * P)
        Y = Y.transpose(3, 0, 1, 2).reshape(F, -1)
        
        # Interpolate along each column (each time series of shape (frame,))
        for i in range(Y.shape[1]):
            y = Y[:, i]
            
            # Get non-NaN indices
            x = np.flatnonzero(~np.isnan(y))
            if len(x) < 2:
                # Not enough data to interpolate
                continue
            
            # Interpolate internal NaNs
            f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)
            xq = np.flatnonzero(np.isnan(y))
            y[xq] = f(xq)
            
            # Fill leading/trailing NaNs
            mask = np.isnan(y)
            if mask.any() and (~mask).any():
                y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])
            
            Y[:, i] = y  # Save back
        
        # Reshape back to original shape: (frame, M, C, P) → (M, C, P, F)
        Y = Y.reshape(F, M, C, P).transpose(1, 2, 3, 0)
        
        return Y
    
    def returnNoseLocs(self):
        return self.data[:, :, 0, :]
    
    def return_lEarLocs(self):
        return self.data[:, :, 1, :]
    
    def return_rEarLocs(self):
        return self.data[:, :, 2, :]
    
    def return_HBLocs(self):
        return self.data[:, :, 3, :]
    
    def return_TBLocs(self):
        return self.data[:, :, 4, :]
    
    def returnGazeVector(self, mouseID):
        """
        Return a normalized and scaled gaze vector (from head base to nose) 
        for all frames for the specified mouse.
        
        Parameters:
            mouseID (int): 0 or 1, indicating which mouse to analyze
            length (float): the desired magnitude of the returned gaze vector
            
        Returns:
            np.ndarray: array of shape (2, num_frames) with extended-length gaze vectors
        """
        length = self.vectorLength
        
        HB = self.data[mouseID, :, self.HB_INDEX, :]  # shape (2, num_frames)
        nose = self.data[mouseID, :, self.NOSE_INDEX, :]
        raw_vec = nose - HB
        norms = np.linalg.norm(raw_vec, axis=0) + 1e-8  # Avoid divide-by-zero
        normalized = raw_vec / norms
        scaled = normalized * length
        return scaled  # shape (2, num_frames)
        
    
    def _point_to_segment_distance(self, point, seg_start, seg_end):
        """Return distance from point to line segment (seg_start -> seg_end)."""
        line_vec = seg_end - seg_start
        pnt_vec = point - seg_start
        line_len = np.dot(line_vec, line_vec)
        if line_len == 0:
            return np.linalg.norm(pnt_vec)
        proj = np.dot(pnt_vec, line_vec) / line_len
        proj = np.clip(proj, 0, 1)
        closest = seg_start + proj * line_vec
        return np.linalg.norm(point - closest)
    
    def _gaze_intersects_body(self, gaze_origin, gaze_vector, target_body):
        """
        Check if the gaze vector intersects the polygonal region defined by:
        left ear → nose → right ear → tail base → left ear.
        
        Parameters:
            gaze_origin (np.ndarray): shape (2,), origin of the gaze vector.
            gaze_vector (np.ndarray): shape (2,), direction of the gaze.
            target_body (np.ndarray): shape (2, 5), [x,y] coordinates of 5 body parts.
            gaze_length (float): length to extend the gaze line.
        
        Returns:
            bool: True if gaze vector intersects the body polygon, else False.
        """
        gaze_length = self.vectorLength
        
        # Indices for relevant parts
        earL = target_body[:, self.earL_INDEX]
        nose = target_body[:, self.NOSE_INDEX]
        earR = target_body[:, self.earR_INDEX]
        tail = target_body[:, self.TB_INDEX]
        
        # Construct polygon path
        polygon_points = [tuple(earL), tuple(nose), tuple(earR), tuple(tail), tuple(earL)]
        body_poly = Polygon(polygon_points)
    
        # Extend the gaze vector in both directions
        gaze_dir = gaze_vector / (np.linalg.norm(gaze_vector) + 1e-8)
        p1 = gaze_origin - gaze_length * gaze_dir
        p2 = gaze_origin + gaze_length * gaze_dir
        gaze_line = LineString([tuple(p1), tuple(p2)])
    
        return gaze_line.intersects(body_poly)
    
    def returnIsStill(self, mouseID, alternateDef = True):
        #Determines whether a mouse has been still for self.minFramesStill frames where stillness is quantified by each body part being within a circle of radius self.stillnessRange for the entire frameCount
        
        if (alternateDef == False):
            """
            Return a boolean array of shape (num_frames,) where True indicates the mouse was still
            for the last `minFramesStill` frames.
            Stillness is defined by each body part remaining within a circle of radius `stillnessRange`.
            """
            num_frames = self.data.shape[-1]
            body_part_positions = self.data[mouseID]  # shape (2, 5, num_frames)
            still_mask = np.zeros(num_frames, dtype=bool)
    
            for t in range(self.minFramesStill, num_frames):
                window = body_part_positions[:, :, t - self.minFramesStill:t]  # shape (2, 5, window)
                # Calculate std dev in x and y for each part over the window
                is_still = True
                for part in range(5):
                    x_std = np.std(window[0, part, :])
                    y_std = np.std(window[1, part, :])
                    if np.sqrt(x_std**2 + y_std**2) > self.stillnessRange:
                        is_still = False
                        break
                if is_still:
                    still_mask[t] = True
    
            return still_mask
        else:
            # Alternate definition: gaze intersects body for minFramesStill consecutive frames
            num_frames = self.data.shape[-1]
            still_mask = np.zeros(num_frames, dtype=bool)
    
            other_mouse = 1 - mouseID
            for t in range(self.minFramesStill, num_frames):
                intersected_all = True
    
                for tau in range(t - self.minFramesStill, t):
                    # Get the gaze origin and direction
                    gaze_origin = self.data[mouseID, :, self.HB_INDEX, tau]       # shape (2,)
                    gaze_vector = self.returnGazeVector(mouseID)[:, tau]          # shape (2,)
                    
                    # Get the target body (2, 5) for the other mouse at frame tau
                    target_body = self.data[other_mouse, :, :, tau]               # shape (2, 5)
    
                    # Check for intersection
                    if not self._gaze_intersects_body(gaze_origin, gaze_vector, target_body):
                        intersected_all = False
                        break
    
                if intersected_all:
                    still_mask[t] = True
    
            return still_mask
            
    
        # Construct polygon path
        polygon_points = [tuple(earL), tuple(nose), tuple(earR), tuple(tail), tuple(earL)]
        body_poly = Polygon(polygon_points)
    
        # Extend the gaze vector in both directions
        gaze_dir = gaze_vector / (np.linalg.norm(gaze_vector) + 1e-8)
        p1 = gaze_origin - gaze_length * gaze_dir
        p2 = gaze_origin + gaze_length * gaze_dir
        gaze_line = LineString([tuple(p1), tuple(p2)])
    
        return gaze_line.intersects(body_poly)
    
    
    def returnIsGazing(self, mouseID, test = False):
        #determine whether mouse mouseID is gazing at the other mouse where a gaze is defined by a mouse standingStill for the self.minFramesStill and the gazeVector passing through the body of the other mouse (estimate of the body using the 5 body parts tracked)  
        """
        Return boolean array where True means that mouseID is still and gazing at the other mouse.
        Gaze is defined as being still and the gaze vector passing near (within threshold)
        any body part of the other mouse.
        """
        num_frames = self.data.shape[-1]
        result = np.zeros(num_frames, dtype=bool)

        still_mask = self.returnIsStill(mouseID)
        gaze_vector = self.returnGazeVector(mouseID)
        HB = self.data[mouseID, :, self.HB_INDEX, :]
        otherID = 1 - mouseID
        other_body = self.data[otherID]  # shape (2, 5, num_frames)

        for t in range(num_frames):
            if not still_mask[t]:
                continue
            gaze_vec = gaze_vector[:, t]
            gaze_origin = HB[:, t]
            target = other_body[:, :, t]  # shape (2, 5)
            if self._gaze_intersects_body(gaze_origin, gaze_vec, target):
                result[t] = True

        return np.where(result)[0] if test else result
    
    
    #Graph Stuff
    
    def returnMouseLocation(self, mouseID):
        """
        Returns a list of strings indicating the region ('lev', 'mid', 'mag') the headbase of the mouse is in for each frame.
        """
        HB_x = self.data[mouseID, 0, self.HB_INDEX, :]  # x-coordinate of headbase across frames
        locations = []
    
        for x in HB_x:
            if self.levRegion[0] <= x < self.levRegion[1]:
                locations.append("lev")
            elif self.middleRegion[0] <= x < self.middleRegion[1]:
                locations.append("mid")
            elif self.magRegion[0] <= x <= self.magRegion[1]:
                locations.append("mag")
            else:
                locations.append("unknown")
    
        return locations
    
    def returnGazeAlignmentHistogram(self, mouseID):
        """
        Computes a histogram of angles (in degrees) between the gaze vector and the 
        tailbase-to-headbase vector, binned in 5-degree intervals.
        
        Returns:
            np.ndarray: array of shape (36,) with counts for 0-5, 6-10, ..., 175-180 degree bins.
        """
        num_frames = self.data.shape[-1]
        
        # Tailbase-to-headbase vector
        TB = self.data[mouseID, :, self.TB_INDEX, :]  # shape (2, num_frames)
        HB = self.data[mouseID, :, self.HB_INDEX, :]  # shape (2, num_frames)
        body_vec = HB - TB                             # shape (2, num_frames)
    
        # Gaze vector
        gaze_vec = self.returnGazeVector(mouseID)      # shape (2, num_frames)
    
        # Normalize vectors
        body_norm = np.linalg.norm(body_vec, axis=0) + 1e-8
        gaze_norm = np.linalg.norm(gaze_vec, axis=0) + 1e-8
        body_unit = body_vec / body_norm
        gaze_unit = gaze_vec / gaze_norm
    
        # Compute dot product
        dot_product = np.sum(body_unit * gaze_unit, axis=0)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # Avoid rounding errors
    
        # Convert to angles in degrees
        angles = np.degrees(np.arccos(dot_product))  # shape (num_frames,)
    
        # Bin the angles into 5-degree intervals (0–180 → 36 bins)
        bins = np.arange(0, 185, 5)  # [0, 5, ..., 180]
        hist, _ = np.histogram(angles, bins=bins)
    
        return hist
    
    def returnInterMouseDistance(self):
        """
        Returns an array of distances between the headbases of the two mice
        for each frame.
    
        Returns:
            np.ndarray: Array of shape (num_frames,) with distances per frame.
        """
        HB_mouse0 = self.data[0, :, self.HB_INDEX, :]  # shape (2, num_frames)
        HB_mouse1 = self.data[1, :, self.HB_INDEX, :]  # shape (2, num_frames)
    
        # Euclidean distance between corresponding frames
        distances = np.linalg.norm(HB_mouse0 - HB_mouse1, axis=0)
        return distances
    
    def returnNumGazeEvents(self, mouseID):
        #A gaze event is a single collection of frames where the mice are gazing. There has to be at least a 5 frame separation with no gazing between gaze events
        
        numGazeEvents = 0
        isGazing = self.returnIsGazing(mouseID)
        lastGaze = -5
        
        for i, frame in enumerate(isGazing):
            if (frame):
                if (i - lastGaze >= 5):
                    numGazeEvents += 1
                lastGaze = i
                
        return numGazeEvents
    
    def returnTotalFramesGazing(self, mouseID):
        g0 = self.returnIsGazing(mouseID)
        totalFramesGazing = np.sum(g0)
        return totalFramesGazing
        
    
    def returnAverageGazeLength(self, mouseID):
        return (self.returnTotalFramesGazing(mouseID) / self.returnNumGazeEvents(mouseID))
    
    def returnNumFrames(self):
        return self.data.shape[-1]
        
        

def visualize_gaze_overlay(
    video_path,
    loader,
    mouseID=0,
    save_path="output.mp4",
    start_frame=0,
    max_frames=300,
    gaze_length=250
):
    '''
    Visual Representation of Gaze Events
        Definitions
            A mouse is considering gazing if it is still and is looking at the other mouse
            
            Stillness is defined as whether all 5 points tracked over {minFramesStill} frames, the standard deviation in the x and y direction is under a threshold {stillnessRange} 
            
            Looking is defined as whether the vector originating from the headbase in the direction 
            of the nose with a length of {vectorLength} intersects with the body made by all the tracked 
            points of the other mouse
    
        Parameters
            a. {stillnessRange}
            b. {minFramesStill}
            c. {vectorLength}
    '''
    
    print("Start")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Could not open video file.")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("width: ", width)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("height: ", height)

    # Temp directory to store images
    temp_dir = Path("temp_gaze_frames")
    temp_dir.mkdir(exist_ok=True)

    gaze_vector = loader.returnGazeVector(mouseID)
    HB = loader.data[mouseID, :, loader.HB_INDEX, :]
    other_body = loader.data[1 - mouseID]  # shape (2, 5, num_frames)
    is_still = loader.returnIsStill(mouseID)
    mouse_region = loader.returnMouseLocation(mouseID)
    other_region = loader.returnMouseLocation(1 - mouseID)
    
    frame_idx = start_frame
    frame_count = 0
    
    while cap.isOpened() and frame_idx < num_frames and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret or frame_idx >= loader.data.shape[-1]:
            break

        gaze_vec = gaze_vector[:, frame_idx]
        gaze_origin = HB[:, frame_idx]
        target = other_body[:, :, frame_idx]

        gaze_dir = gaze_vec / (np.linalg.norm(gaze_vec) + 1e-8)
        gaze_tip = gaze_origin + gaze_length * gaze_dir

        p1 = tuple(np.round(gaze_origin).astype(int))
        p2 = tuple(np.round(gaze_tip).astype(int))

        intersect = loader._gaze_intersects_body(gaze_origin, gaze_vec, target)
        still = is_still[frame_idx]
        gazing = intersect and still

        # Set line color based on state
        if gazing:
            color = (0, 0, 255)  # Red
        elif still:
            color = (255, 0, 0)  # Blue
        else:
            color = (0, 255, 0)  # Green

        # Draw the gaze vector
        cv2.line(frame, p1, p2, color, 2)
        
        #Draw Stillness Circles        
        #for part_idx in range(5):
                #cx, cy = loader.data[mouseID, :, part_idx, frame_idx].astype(int)
                #cv2.circle(frame, (cx, cy), int(loader.stillnessRange), (255, 255, 0), 1)


        cv2.putText(frame, f"Intersecting: {intersect}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Gazing: {gazing}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Still: {still}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw zone boundaries
        cv2.line(frame, (loader.levBoundary, 0), (loader.levBoundary, height), (255, 255, 0), 2)   # Cyan line for levBoundary
        cv2.line(frame, (loader.magBoundary, 0), (loader.magBoundary, height), (255, 0, 255), 2)   # Magenta line for magBoundary

        # Optional: label the zones
        cv2.putText(frame, "Lev", (loader.levBoundary//2 - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        cv2.putText(frame, "Mid", (width//2 - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
        cv2.putText(frame, "Mag", (loader.magBoundary + (width - loader.magBoundary)//2 - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)        
        
        #Mouse Region
        region_self = mouse_region[frame_idx]
        region_other = other_region[frame_idx]
        
        cv2.putText(frame, f"Mouse{mouseID}: {region_self}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Mouse{1 - mouseID}: {region_other}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw body polygon
        polygon_indices = [
            loader.earL_INDEX,
            loader.NOSE_INDEX,
            loader.earR_INDEX,
            loader.TB_INDEX
        ]
        polygon_points = [tuple(np.round(target[:, idx]).astype(int)) for idx in polygon_indices]
        polygon_points.append(polygon_points[0])  # close the loop

        for j in range(len(polygon_points) - 1):
            cv2.line(frame, polygon_points[j], polygon_points[j+1], (128, 128, 128), 1)

        frame_filename = temp_dir / f"frame_{frame_count:05d}.png"
        cv2.imwrite(str(frame_filename), frame)
        print (f"Frame {frame_count}")
        
        frame_idx += 1
        frame_count += 1

    cap.release()

    # Build the video using ffmpeg
    print(f"Saving video to {save_path} using ffmpeg...")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(int(fps)),
        "-i", str(temp_dir / "frame_%05d.png"),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        str(save_path)
    ]
    result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("FFmpeg error:", result.stderr.decode())
        raise RuntimeError("FFmpeg failed to create video.")

    shutil.rmtree(temp_dir)
    print(f"Video saved to {save_path}")
    
 
h5_file = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5"
video_file = "/Users/david/Downloads/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.mp4"    
 
#h5_file = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5"
#video_file = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.mp4"    


#loader = posLoader(h5_file)
#visualize_gaze_overlay(video_file, loader, mouseID=0, save_path = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Graphs/Videos/testGazeVid_noFillMissingData.mp4")

    
    
    