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
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
from mag_class import magLoader
from lev_class import levLoader


class posLoader:
    
    def __init__(self, filename, totalFrames = 1000):
        
        self.filename = filename
        with h5py.File(filename, "r") as f:
            if 'tracks' not in f:
                print("Dataset 'tracks' not found in file")
                return
            raw_data = f['tracks'][:]  # Shape: (2, 2, 5, num_frames)

            # Calculate NaN percentage
            total_values = np.prod(raw_data.shape)
            nan_count = np.isnan(raw_data).sum()
            nan_percentage = 100 * nan_count / total_values
            #print(f"[{filename}] Missing data: {nan_count} NaNs out of {total_values} values " f"({nan_percentage:.2f}%) before interpolation")
    
            self.data = self._fill_missing_data(raw_data) #Shape: (2, 2, 5, x) --> (rat0/1, x/y, which_body_part, which_frame)
        
        self.totalFrames = totalFrames
        
        self.minFramesStill = 10
        self.stillnessRange = 10
        
        #List of body parts tracked
        self.NOSE_INDEX = 0
        self.earL_INDEX = 1
        self.earR_INDEX = 2
        self.HB_INDEX = 3 #head base index
        self.TB_INDEX = 4 #tail base index
        
        self.vectorLength = 425
        
        #Frame Dimensions
        self.width = 1392
        self.height = 640
        
        # Define horizontal boundaries for zones
        self.levBoundary = 351
        self.magBoundary = 1041
        
        #Subdivide Lev and Mag Zones
        self.levTopTR = (350, 10)
        self.levTopBL = (10, 310)
        
        self.levBotTR = (350, 330)
        self.levBotBL = (10, 630)
        
        self.magTopTR = (1382, 10)
        self.magTopBL = (1042, 310)
        
        self.magBotTR = (1382, 330)
        self.magBotBL = (1042, 630)
        
        self.topWall = 80
        self.bottomWall = 580
        
        # Define regions based on x-coordinate of headbase
        self.levRegion = (0, self.levBoundary)
        self.middleRegion = (self.levBoundary, self.magBoundary)
        self.magRegion = (self.magBoundary, self.width)
        
        self.possibleLocations = ['lev_top', 'lev_bottom', 'mag_top', 'mag_bottom', 'mid', 'other']
        
        
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
    
    def returnGazeVector(self, mouseID, minDist=150):
        """
        Return a normalized gaze vector (head base → nose) scaled between minDist and vectorLength.
        
        The vector represents the line segment along which we check gaze, starting at minDist away 
        from the head base and extending out to vectorLength.
        """
        length = self.vectorLength  # e.g. 400
    
        HB = self.data[mouseID, :, self.HB_INDEX, :]   # (2, num_frames)
        nose = self.data[mouseID, :, self.NOSE_INDEX, :]
        raw_vec = nose - HB
        norms = np.linalg.norm(raw_vec, axis=0) + 1e-9
        normalized = raw_vec / norms  # direction only
    
        # Scale to full length
        scaled = normalized * length   # end point direction
    
        return scaled
        
    
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
    
    def _gaze_intersects_body(self, gaze_origin, gaze_vector, target_body, minDist=150):
        """
        Check if gaze vector intersects target body polygon, starting minDist away 
        from gaze_origin and extending to vectorLength.
        
        Parameters:
            gaze_origin (np.ndarray): shape (2,), origin of the gaze vector.
            gaze_vector (np.ndarray): shape (2,), direction of the gaze.
            target_body (np.ndarray): shape (2, 5), [x,y] coordinates of 5 body parts.
            gaze_length (float): length to extend the gaze line.
            minDist: offset distance away to separate interactions and gazing. 
        """
        gaze_length = self.vectorLength
    
        # Polygon for the body
        earL = target_body[:, self.earL_INDEX]
        nose = target_body[:, self.NOSE_INDEX]
        earR = target_body[:, self.earR_INDEX]
        tail = target_body[:, self.TB_INDEX]
        polygon_points = [tuple(earL), tuple(nose), tuple(earR), tuple(tail), tuple(earL)]
        body_poly = Polygon(polygon_points)
    
        # Normalize gaze direction
        gaze_dir = gaze_vector / (np.linalg.norm(gaze_vector) + 1e-8)
    
        # Offset start point by minDist
        p1 = gaze_origin + minDist * gaze_dir
        p2 = gaze_origin + gaze_length * gaze_dir
        gaze_line = LineString([tuple(p1), tuple(p2)])

        return gaze_line.intersects(body_poly)
    
    def returnIsStill(self, mouseID, alternateDef = True, minDist = 150):
        #Determines whether a mouse has been still for self.minFramesStill frames where stillness is quantified by each body part being within a circle of radius self.stillnessRange for the entire frameCount
        
        if (alternateDef == False):
            """
            Return a boolean array of shape (num_frames,) where True indicates the mouse was still
            for the last `minFramesStill` frames.
            Stillness is defined by each body part having a standard deviation over the {minFramesStill} 
            less than `stillnessRange`.
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
            '''
            Alternate definition: gaze intersects body for minFramesStill consecutive frames
            '''
            num_frames = self.data.shape[-1]
            #print("num_frames (isStill): ", num_frames)
            still_mask = np.zeros(num_frames, dtype=bool)
    
            other_mouse = 1 - mouseID
            for t in range(self.minFramesStill, num_frames):
                #print("t: ", t)
                intersected_all = True
    
                for tau in range(t - self.minFramesStill, t):
                    # Get the gaze origin and direction
                    gaze_origin = self.data[mouseID, :, self.HB_INDEX, tau]       # shape (2,)
                    gaze_vector = self.returnGazeVector(mouseID)[:, tau]          # shape (2,)
                    
                    # Get the target body (2, 5) for the other mouse at frame tau
                    target_body = self.data[other_mouse, :, :, tau]               # shape (2, 5)
    
                    # Check for intersection
                    if not self._gaze_intersects_body(gaze_origin, gaze_vector, target_body, minDist=minDist):
                        intersected_all = False
                        break
    
                if intersected_all:
                    still_mask[t] = True
    
            return still_mask
    
    
    def returnIsGazing(self, mouseID, test = False, alternateDef = True, minDist=150):
        #determine whether mouse mouseID is gazing at the other mouse where a gaze is defined by a mouse standingStill for the self.minFramesStill and the gazeVector passing through the body of the other mouse (estimate of the body using the 5 body parts tracked)  
        """
        Return boolean array where True means that mouseID is still and gazing at the other mouse.
        Gaze is defined as being still and the gaze vector passing near (within threshold)
        any body part of the other mouse.
        """
        num_frames = self.data.shape[-1]
        #print("num_frames is: ", num_frames)
        result = np.zeros(num_frames, dtype=bool)

        still_mask = self.returnIsStill(mouseID, alternateDef, minDist=minDist) #'still' aka gazing if the last 10 frames have intersected the other rat. However, 
        gaze_vector = self.returnGazeVector(mouseID)
        HB = self.data[mouseID, :, self.HB_INDEX, :]
        otherID = 1 - mouseID
        other_body = self.data[otherID]  # shape (2, 5, num_frames)

        #print("numFrames")
    
        for t in range(num_frames):
            #print("t: ", t)
            if not still_mask[t]:
                continue
            gaze_vec = gaze_vector[:, t]
            gaze_origin = HB[:, t]
            target = other_body[:, :, t]  # shape (2, 5)
            if self._gaze_intersects_body(gaze_origin, gaze_vec, target, minDist=minDist):
                result[t] = True
        
        isInteraction = self.returnIsInteracting()
        if (len(result) != len(isInteraction)):
            print("LENGTHS NOT EQUAL")
        
        result = np.array(result, dtype=bool)
        isInteraction = np.array(isInteraction, dtype=bool)
        filteredGaze = result & (~isInteraction)
        
        isGazing = np.where(result)[0] if test else filteredGaze
        
        return result
    
    
    
    #Gazing at Lev or Mag Code
    def _gaze_intersects_rect(self, gaze_origin, gaze_vector, rectBL, rectTR, useMinDist=True, minDist=150):
        """
        Check if a gaze vector intersects a rectangle (lever or magazine).
        
        Parameters:
            gaze_origin (np.ndarray): (2,), origin of the gaze vector
            gaze_vector (np.ndarray): (2,), direction of the gaze
            rectBL (tuple): bottom-left (x, y) corner of rectangle
            rectTR (tuple): top-right (x, y) corner of rectangle
            useMinDist (bool): if True, apply minDist offset to gaze origin
            minDist (float): distance offset for gaze start
        """
        gaze_length = self.vectorLength
    
        # Rectangle polygon
        x1, y1 = rectBL
        x2, y2 = rectTR
        rect_poly = Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
    
        # Normalize gaze direction
        gaze_dir = gaze_vector / (np.linalg.norm(gaze_vector) + 1e-8)
    
        # Offset start
        offset = minDist if useMinDist else 0
        p1 = gaze_origin + offset * gaze_dir
        p2 = gaze_origin + gaze_length * gaze_dir
        gaze_line = LineString([tuple(p1), tuple(p2)])
    
        return gaze_line.intersects(rect_poly)
    
    def returnIsLookingAtObjects(self, ratID, target="lever", useMinDist=True, returnToporBottom=True):
        """
        Return boolean arrays indicating whether mouseID is gazing/interacting 
        with each lever or magazine across frames.
    
        Parameters:
            mouseID (int): which mouse (0 or 1)
            target (str): "lever" or "mag"
            useMinDist (bool): if True, check gazing (with threshold); 
                               if False, check interacting (no threshold)
    
        Returns:
            (bool arrays): 4 arrays (per object), each shape (num_frames,)
        """
    
        num_frames = self.data.shape[-1]
        gaze_vector = self.returnGazeVector(ratID)
        HB = self.data[ratID, :, self.HB_INDEX, :]  # (2, num_frames)
    
        # Lever/mag positions
        lever_rects = [((100, 240), (160, 180)),  # Top lever
                       ((100, 415), (160, 475))]  # Bottom lever
        mag_rects = [((1240, 245), (1300, 185)),  # Top mag
                     ((1240, 415), (1300, 475))]  # Bottom mag
    
        rects = lever_rects if target == "lever" else mag_rects
    
        # Track intersections frame-by-frame
        top_hits = np.zeros(num_frames, dtype=bool)
        bot_hits = np.zeros(num_frames, dtype=bool)
    
        for t in range(num_frames):
            gvec = gaze_vector[:, t]
            gorigin = HB[:, t]
    
            # Top rectangle
            if self._gaze_intersects_rect(gorigin, gvec, rects[0][0], rects[0][1], useMinDist):
                top_hits[t] = True
    
            # Bottom rectangle
            if self._gaze_intersects_rect(gorigin, gvec, rects[1][0], rects[1][1], useMinDist):
                bot_hits[t] = True
    
        # Enforce self.minFramesStill consecutive hits
        res1 = np.zeros(num_frames, dtype=bool)
        res2 = np.zeros(num_frames, dtype=bool)
    
        for t in range(self.minFramesStill, num_frames):
            if np.all(top_hits[t - self.minFramesStill:t]):
                res1[t] = True
            if np.all(bot_hits[t - self.minFramesStill:t]):
                res2[t] = True
        
        if (returnToporBottom):
            res = np.logical_or(res1, res2)
            return res
        else:
            return res1, res2
        
    #Graph Stuff
    
    def returnMouseLocation(self, ratID):
        """
        Returns a list of strings indicating the region of the mouse's headbase for each frame.
        Possible regions: 'lev_top', 'lev_bottom', 'mag_top', 'mag_bottom', 'mid', 'other'
        """
        HB_x = self.data[ratID, 0, self.HB_INDEX]  # x-coordinate of headbase
        HB_y = self.data[ratID, 1, self.HB_INDEX]  # y-coordinate of headbase
        locations = []
    
        for x, y in zip(HB_x, HB_y):
            if self.levTopBL[0] <= x <= self.levTopTR[0] and self.levTopTR[1] <= y <= self.levTopBL[1]:
                locations.append("lev_top")
            elif self.levBotBL[0] <= x <= self.levBotTR[0] and self.levBotTR[1] <= y <= self.levBotBL[1]:
                locations.append("lev_bottom")
            elif self.magTopBL[0] <= x <= self.magTopTR[0] and self.magTopTR[1] <= y <= self.magTopBL[1]:
                locations.append("mag_top")
            elif self.magBotBL[0] <= x <= self.magBotTR[0] and self.magBotTR[1] <= y <= self.magBotBL[1]:
                locations.append("mag_bottom")
            elif self.levBoundary <= x < self.magBoundary:
                locations.append("mid")
            else:
                locations.append("other")
    
        return locations
    
    def returnRatLocationTime(self, ratID, t):
        """
        Returns a list of strings indicating the region of the mouse's headbase for each frame.
        Possible regions: 'lev_top', 'lev_bottom', 'mag_top', 'mag_bottom', 'mid', 'other'
        """
        x = self.data[ratID, 0, self.HB_INDEX, t]  # x-coordinate of headbase
        y = self.data[ratID, 1, self.HB_INDEX, t]  # y-coordinate of headbase
    
        if self.levTopBL[0] <= x <= self.levTopTR[0] and self.levTopTR[1] <= y <= self.levTopBL[1]:
            locations = ("lev_top")
        elif self.levBotBL[0] <= x <= self.levBotTR[0] and self.levBotTR[1] <= y <= self.levBotBL[1]:
            locations = ("lev_bottom")
        elif self.magTopBL[0] <= x <= self.magTopTR[0] and self.magTopTR[1] <= y <= self.magTopBL[1]:
            locations = ("mag_top")
        elif self.magBotBL[0] <= x <= self.magBotTR[0] and self.magBotTR[1] <= y <= self.magBotBL[1]:
            locations = ("mag_bottom")
        elif self.levBoundary <= x < self.magBoundary:
            locations = ("mid")
        else:
            locations = ("other")
    
        return locations
    
    def returnRatHBPosition(self, ratID, t):
        x = self.data[ratID, 0, self.HB_INDEX, t]  # x-coordinate of headbase
        y = self.data[ratID, 1, self.HB_INDEX, t]  # y-coordinate of headbase
        
        return(x, y)
        
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
    
    def returnNumGazeEvents(self, mouseID, alternateDef = True):
        #A gaze event is a single collection of frames where the mice are gazing. There has to be at least a 5 frame separation with no gazing between gaze events
        
        numGazeEvents = 0
        isGazing = self.returnIsGazing(mouseID, alternateDef = alternateDef)
        lastGaze = -5
        
        for i, frame in enumerate(isGazing):
            if (frame):
                if (i - lastGaze >= 5):
                    numGazeEvents += 1
                lastGaze = i
                
        return numGazeEvents
    
    def returnTotalFramesGazing(self, mouseID, alternateDef = True):
        g0 = self.returnIsGazing(mouseID, alternateDef = alternateDef)
        #print("isGazingArr: ", g0)
        totalFramesGazing = np.sum(g0)
        return totalFramesGazing
        
    def returnTotalFramesInteracting(self):
        interacting = self.returnIsInteracting()
        totalFramesInteracting = np.sum(interacting)
        print(interacting[:10])
        print("sum: ", np.sum(interacting[:10]))
        return totalFramesInteracting
    
    def returnAverageGazeLength(self, mouseID):
        return (self.returnTotalFramesGazing(mouseID) / self.returnNumGazeEvents(mouseID))
    
    def returnNumFrames(self):
        return self.data.shape[-1]
    
    def checkSelfIntersection(self, ratID):
        """
        For each frame, determines whether the polygon formed by
        nose → right ear → tailbase → left ear → nose intersects itself.
        Returns a boolean array of shape (num_frames,) where True = self-intersecting.
        """
        def segments_intersect(p1, p2, q1, q2):
            """Check if line segments p1-p2 and q1-q2 intersect."""
            def ccw(a, b, c):
                # Compute the cross product (b-a) x (c-a)
                return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
        
            # Check if the orientations differ for the endpoints of each segment
            o1 = ccw(p1, q1, q2)
            o2 = ccw(p2, q1, q2)
            o3 = ccw(p1, p2, q1)
            o4 = ccw(p1, p2, q2)
        
            # General case: segments intersect if orientations of endpoints differ
            if o1 * o2 < 0 and o3 * o4 < 0:
                return True
        
            # Special cases: handle collinear segments
            def on_segment(p, q, r):
                return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                        q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
        
            if o1 == 0 and on_segment(p1, q1, p2): return True
            if o2 == 0 and on_segment(p1, q2, p2): return True
            if o3 == 0 and on_segment(q1, p1, q2): return True
            if o4 == 0 and on_segment(q1, p2, q2): return True
        
            return False
    
        data = self.data  # shape: (2, 2, 5, num_frames)
        num_frames = data.shape[-1]
        result = []
    
        for f in range(num_frames):
            # Extract points in order: nose → right ear → tailbase → left ear → nose
            nose = data[ratID, :, self.NOSE_INDEX, f]
            right_ear = data[ratID, :, self.earR_INDEX, f]
            tailbase = data[ratID, :, self.TB_INDEX, f]
            left_ear = data[ratID, :, self.earL_INDEX, f]
    
            pts = [tuple(nose), tuple(right_ear), tuple(tailbase), tuple(left_ear)]
            pts.append(pts[0])  # close the polygon
            edges = [(pts[i], pts[i+1]) for i in range(4)]
            #print("Edges: ", edges)
    
            intersect = False
            for i in range(len(edges)):
                for j in range(i + 1, len(edges)):
                    # Skip if edges share a point (i.e., adjacent edges)
                    if set(edges[i]) & set(edges[j]):
                        continue
                    if segments_intersect(edges[i][0], edges[i][1], edges[j][0], edges[j][1]):
                        intersect = True
                        break
                if intersect:
                    break
    
            result.append(intersect)
    
        return result  # Boolean array, True = frame has self-intersecting shape
    
    def returnNumFramesSelfIntersection(self, ratID):
        lst = self.checkSelfIntersection(ratID)
        true_count = lst.count(True)
        return true_count
    
    def returnStandardizedDistanceMoved(self, ratID):
        x_coords = self.data[ratID, 0, self.HB_INDEX, :]
        #print("x_coords")
        #print(x_coords)
        y_coords = self.data[ratID, 1, self.HB_INDEX, :]
        #print("y_coords: ")
        #print(y_coords)

        # Calculate total distance as sum of Euclidean distances between consecutive frames
        valid_indices = ~np.isnan(x_coords) & ~np.isnan(y_coords)
        x = x_coords[valid_indices]
        y = y_coords[valid_indices]

        dx = np.diff(x)
        #print("dx: ", dx)
        dy = np.diff(y)
        #print("dy: ", dy)
        
        dist = np.sqrt(dx ** 2 + dy ** 2)
        total_distance = np.sum(dist)
        
        #total_distance = np.sum(totDist)
        
        if (self.totalFrames > 0):
            #print("Result: ", total_distance / self.totalFrames)
            return total_distance / self.totalFrames
        else:
            return 0
    
    def getHeadBodyTrajectory(self, rat_id):
        """
        Returns: ndarray of shape (2, num_frames), where each column is the headbase (x, y) position.
        """
        return self.data[rat_id, :, self.HB_INDEX, :]
    
    def computeVelocity(self, rat_id):
        """
        Returns: ndarray of shape (num_frames,), velocity (pixels/frame) of the headbase.
        """
        pos = self.getHeadBodyTrajectory(rat_id)  # shape: (2, num_frames)
        diffs = np.diff(pos, axis=1)
        speeds = np.linalg.norm(diffs, axis=0)
        speeds = np.concatenate(([0], speeds))  # Add zero velocity for the first frame
        return speeds

    def getLeverZone(self, rat_id):
        """
        Returns a boolean array where True indicates the rat is in the lever zone.
        """
        locations = self.returnMouseLocation(rat_id)
        return np.array([loc.startswith("lev") for loc in locations])

    def getRewardZone(self, rat_id):
        """
        Returns a boolean array where True indicates the rat is in the reward zone.
        """
        locations = self.returnMouseLocation(rat_id)
        return np.array([loc.startswith("mag") for loc in locations])
    
    def approachingLever(self, rat_id, t):
        """
        Returns True if the rat is facing the lever (left side) and its headbase is in
        'lev_top', 'lev_bottom', or 'mid' regions at frame t.
        
        Args:
            rat_id (int): 0 or 1, indicating which rat.
            t (int): Frame index.
        
        Returns:
            bool: True if rat is facing left and in specified regions, False otherwise.
        """
        # Check location
        locations = self.returnMouseLocation(rat_id)
        if t >= len(locations) or locations[t] not in ['lev_top', 'lev_bottom', 'mid']:
            return False
        
        # Get headbase and nose positions
        headbase = self.data[rat_id, :, self.HB_INDEX, t]  # Shape: (2,)
        nose = self.data[rat_id, :, self.NOSE_INDEX, t]    # Shape: (2,)
        
        # Check for NaN values
        if np.any(np.isnan(headbase)) or np.any(np.isnan(nose)):
            return False
        
        # Calculate gaze vector (nose - headbase)
        gaze_vector = nose - headbase  # Shape: (2,)
        
        # Check if gaze is facing left (negative x-direction)
        # Gaze vector's x-component should be negative and significant
        print("gaze_vector: ", gaze_vector)
        print("gaze_vector[0]: ", gaze_vector[0])
        return gaze_vector[0] < -20  # Small threshold to avoid numerical noise
    
    def approachingMagazine(self, rat_id, t):
        """
        Returns True if the rat is facing the magazine (right side) and its headbase is in
        'mag_top', 'mag_bottom', or 'mid' regions at frame t.
        
        Args:
            rat_id (int): 0 or 1, indicating which rat.
            t (int): Frame index.
        
        Returns:
            bool: True if rat is facing right and in specified regions, False otherwise.
        """
        # Check location
        locations = self.returnMouseLocation(rat_id)
        if t >= len(locations) or locations[t] not in ['mag_top', 'mag_bottom', 'mid']:
            return False
        
        # Get headbase and nose positions
        headbase = self.data[rat_id, :, self.HB_INDEX, t]  # Shape: (2,)
        nose = self.data[rat_id, :, self.NOSE_INDEX, t]    # Shape: (2,)
        
        # Check for NaN values
        if np.any(np.isnan(headbase)) or np.any(np.isnan(nose)):
            return False
        
        # Calculate gaze vector (nose - headbase)
        gaze_vector = nose - headbase  # Shape: (2,)
        
        # Check if gaze is facing right (positive x-direction)
        # Gaze vector's x-component should be positive and significant
        return gaze_vector[0] > 20  # Small threshold to avoid numerical noise
    
    def distanceFromLever(self, rat_id, t):
        '''
        Returns the minimum distance of the rat with id rat_id from either lever 
        at time/frame index `t`. The location of the rat is defined by its headbase.
        
        Assumes:
        - self.data has shape (num_rats, num_coords=2, num_bodyparts, num_frames)
        - self.HB_INDEX is the index corresponding to the headbase location
        '''
    
        # Get headbase coordinates of the specified rat at time t
        headbase = self.data[rat_id, :, self.HB_INDEX, t]  # Shape: (2,)
    
        # Define lever positions
        lever1 = np.array([75, 160])
        lever2 = np.array([75, 480])
    
        # Compute Euclidean distances to each lever
        dist1 = np.linalg.norm(headbase - lever1)
        dist2 = np.linalg.norm(headbase - lever2)
    
        # Return the minimum of the two
        return min(dist1, dist2)
    
    def returnInteractionDistance(self):        
        # Assumes self.data is shape: (2, 2, 5, num_frames)
        num_frames = self.data.shape[3]
        distances = []
    
        for frame in range(num_frames):
            rat_id = 0
            other_id = 1
            
            # Get (x, y) position of nose of rat 0 at this frame
            nose0 = self.data[rat_id, :, self.NOSE_INDEX, frame]
    
            # Get all body parts of rat 1 at this frame
            hb1 = self.data[other_id, :, self.HB_INDEX, frame]
            nose1 = self.data[other_id, :, self.NOSE_INDEX, frame]
            tb1 = self.data[other_id, :, self.TB_INDEX, frame]
            earL1 = self.data[other_id, :, self.earL_INDEX, frame]
            earR1 = self.data[other_id, :, self.earR_INDEX, frame]
    
            # Stack all body parts of rat 1
            body_parts_other = np.stack([hb1, nose1, tb1, earL1, earR1], axis=0)
    
            # Compute Euclidean distances from nose0 to each of rat 1's body parts
            dists = np.linalg.norm(body_parts_other - nose0, axis=1)
    
            # Get minimum distance
            min_dist = np.min(dists)
                
            distances.append(min_dist)
    
        return distances
    
    def distanceFromWall(self, ratID, t):
        headbase_y = self.data[ratID, 1, self.HB_INDEX, t]
        
        return min(abs(headbase_y - self.topWall), abs(headbase_y - self.bottomWall))
    
    def returnIsInteracting(self):
        '''
        List of size totFrames that returns 1 if 
        '''
        
        MIN_FRAMES = 10
        MAX_DIST = 90
        isInteracting = []
        distances = self.returnInteractionDistance()
        count = 0
        numFrames = self.returnNumFrames()
        
        for t in range(numFrames):
            loc0 = self.returnRatLocationTime(0, t)
            loc1 = self.returnRatLocationTime(1, t)
            dist = distances[t]
                        
            if (dist < MAX_DIST and (loc0 != 'other' and loc1 != 'other' and ((loc0 != 'lev_top' or loc1 != 'lev_bottom') and (loc1 != 'lev_top' or loc0 != 'lev_bottom') and (loc0 != 'mag_top' or loc1 != 'mag_bottom') and (loc1 != 'mag_top' or loc0 != 'mag_bottom')))):
                count += 1
                if (count >= MIN_FRAMES):
                    isInteracting.append(1)
                else:
                    isInteracting.append(0)
            else:
                count = 0
                isInteracting.append(0)
                
        return isInteracting
    
    def plot_rat_heatmap(self, savepath="rat_heatmap.png", bodypart=0, bins=100, sigma = 2):
        """
        Create a heatmap of Rat0 (red) vs Rat1 (blue) positions.
        
        Parameters:
            savepath (str): Path to save the heatmap image.
            bodypart (int): Index of body part to plot (default: 0 = headbase).
            bins (int): Resolution of the heatmap bins.
            sigma (float): Gaussian smoothing parameter.
        """
        from scipy.ndimage import gaussian_filter

        #Vars 
        clip_percent=97  #clip_percent (float): Percentile for clipping (e.g. 99 = clip top 1%).
        gamma = 0.2 #Gamma): Gamma correction (<1 = more contrast in low values).
        
        # Extract x, y for each rat
        rat0_x = self.data[0, 0, bodypart, :]
        rat0_y = self.data[0, 1, bodypart, :]
        rat1_x = self.data[1, 0, bodypart, :]
        rat1_y = self.data[1, 1, bodypart, :]
    
        # Remove NaNs
        mask0 = ~np.isnan(rat0_x) & ~np.isnan(rat0_y)
        mask1 = ~np.isnan(rat1_x) & ~np.isnan(rat1_y)
        rat0_x, rat0_y = rat0_x[mask0], rat0_y[mask0]
        rat1_x, rat1_y = rat1_x[mask1], rat1_y[mask1]
    
        # Define bin ranges (shared across both rats)
        all_x = np.concatenate([rat0_x, rat1_x])
        all_y = np.concatenate([rat0_y, rat1_y])
        xedges = np.linspace(all_x.min(), all_x.max(), bins)
        yedges = np.linspace(all_y.min(), all_y.max(), bins)
        
        # Aspect ratio of the arena
        arena_width = all_x.max() - all_x.min()
        arena_height = all_y.max() - all_y.min()
        aspect_ratio = arena_width / arena_height
    
        # Compute 2D histograms
        H0, _, _ = np.histogram2d(rat0_x, rat0_y, bins=[xedges, yedges])
        H1, _, _ = np.histogram2d(rat1_x, rat1_y, bins=[xedges, yedges])
    
        # Smooth with Gaussian filter
        H0 = gaussian_filter(H0, sigma=sigma)
        H1 = gaussian_filter(H1, sigma=sigma)
    
        # Clip extreme values to increase contrast
        vmax0 = np.percentile(H0, clip_percent)
        vmax1 = np.percentile(H1, clip_percent)
        H0 = np.clip(H0, 0, vmax0)
        H1 = np.clip(H1, 0, vmax1)
    
        # Normalize to [0,1]
        H0 = H0 / H0.max() if H0.max() > 0 else H0
        H1 = H1 / H1.max() if H1.max() > 0 else H1
    
        # Apply gamma correction to boost contrast
        H0 = H0 ** gamma
        H1 = H1 ** gamma
    
        # Create RGB heatmap
        '''heatmap_rgb = np.zeros((H0.shape[0], H0.shape[1], 3))
        heatmap_rgb[:, :, 0] = H0  # Red = rat0
        heatmap_rgb[:, :, 2] = H1  # Blue = rat1
    
        # Plot
        plt.figure(figsize=(8, 6))
        plt.imshow(
            np.flipud(heatmap_rgb),
            extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()],
            origin="lower",
            aspect="auto"
        )
        plt.xlabel("X position (px)")
        plt.ylabel("Y position (px)")
        plt.title(f"Rat0 (Red) vs Rat1 (Blue) Heatmap")'''
        
        # Plot with colormaps
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
        im0 = axes[0].imshow(
            np.flipud(H0),
            extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()],
            origin="lower",
            aspect="auto",
            cmap="hot"
        )
        axes[0].set_title("Rat0 Heatmap")
        fig.colorbar(im0, ax=axes[0])
    
        im1 = axes[1].imshow(
            np.flipud(H1),
            extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()],
            origin="lower",
            aspect="auto",
            cmap="hot"
        )
        axes[1].set_title("Rat1 Heatmap")
        fig.colorbar(im1, ax=axes[1])
    
        for ax in axes:
            ax.set_xlabel("X position (px)")
            ax.set_ylabel("Y position (px)")

        plt.tight_layout()
        plt.savefig(savepath, dpi=300)
        plt.show()
        plt.close()
        
        # Function to plot & save with correct proportions
        def save_heatmap(H, title, filename):
            fig_width = 6  # base size in inches
            fig_height = fig_width / aspect_ratio
            plt.figure(figsize=(fig_width, fig_height))
    
            im = plt.imshow(
                np.flipud(H),
                extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()],
                origin="lower",
                aspect="auto",
                cmap="hot"
            )
            plt.colorbar(im, label="Density")
            plt.xlabel("X position (px)")
            plt.ylabel("Y position (px)")
            plt.title(title)
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()
    
        # Save Rat0 and Rat1 heatmaps
        save_heatmap(H0, "Rat0 Heatmap", savepath.replace(".png", "_rat0.png"))
        save_heatmap(H1, "Rat1 Heatmap", savepath.replace(".png", "_rat1.png"))


def visualize_gaze_overlay(
    video_path,
    loader,
    lev,
    mag,
    mouseID=0,
    save_path="output.mp4",
    start_frame=0,
    max_frames=100,
    gaze_length=425, 
    minDist = 150
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
    
    # State-related data
    state_names = ["idle", "approaching lever", "approaching reward", "waiting", "pressed", "reward taken", "exploring", "false mag"]
    pos_data = loader.getHeadBodyTrajectory(mouseID).T  # Shape: (num_frames, 2)
    velocities = loader.computeVelocity(mouseID)
    lever_zone = loader.getLeverZone(mouseID)
    reward_zone = loader.getRewardZone(mouseID)
    press_frames = lev.getLeverPressFrames(mouseID)
    reward_frames = mag.getRewardReceivedFrames(mouseID)
    false_mag_entry = mag.getEnteredMagFrames(mouseID)
    distances = loader.returnInteractionDistance()
    print("Reward_Frames: ", reward_frames)
    
    self_intersecting_rat1 = loader.checkSelfIntersection(1)
    #print("self_intersecting_rat1: ", self_intersecting_rat1)
    
    frame_idx = start_frame
    frame_count = 0
    
    while cap.isOpened() and frame_idx < num_frames and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret or frame_idx >= loader.data.shape[-1]:
            break

        gaze_vec = gaze_vector[:, frame_idx]
        gaze_origin = HB[:, frame_idx]
        target = other_body[:, :, frame_idx]

        # Normalize gaze direction
        gaze_dir = gaze_vec / (np.linalg.norm(gaze_vec) + 1e-8)
    
        # Apply 75px offset for start, 400px for end
        p1 = gaze_origin + minDist * gaze_dir
        p2 = gaze_origin + gaze_length * gaze_dir
    
        p1 = tuple(np.round(p1).astype(int))
        p2 = tuple(np.round(p2).astype(int))


        intersect = loader._gaze_intersects_body(gaze_origin, gaze_dir, target)
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
        
        cv2.putText(frame, f"InterMouseDistance: {distances[frame_idx]}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        
        # Draw self-intersecting status for Rat 1 in bottom right
        status_text = f"Rat1 Self-Intersecting: {self_intersecting_rat1[frame_idx]}"
        text_size, _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_width, text_height = text_size
        bottom_right_x = width - text_width - 50
        bottom_right_y = height - 50
        cv2.putText(frame, status_text, (bottom_right_x, bottom_right_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        

        # Draw zone boundaries
        cv2.line(frame, (loader.levBoundary, 0), (loader.levBoundary, height), (255, 255, 0), 2)   # Cyan line for levBoundary
        cv2.line(frame, (loader.magBoundary, 0), (loader.magBoundary, height), (255, 0, 255), 2)   # Magenta line for magBoundary

        # Optional: label the zones
        #cv2.putText(frame, "Lev", (loader.levBoundary//2 - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        cv2.putText(frame, "Mid", (width//2 - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
        #cv2.putText(frame, "Mag", (loader.magBoundary + (width - loader.magBoundary)//2 - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)        
        
        # Draw wall boundaries
        cv2.line(frame, (0, loader.topWall), (width, loader.topWall), (0, 255, 255), 2)     # Yellow line for topWall
        cv2.line(frame, (0, loader.bottomWall), (width, loader.bottomWall), (0, 165, 255), 2) # Orange line for bottomWall
        
        cv2.putText(frame, "Top Wall", (10, loader.topWall - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, "Bottom Wall", (10, loader.bottomWall + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        #Mouse Region
        region_self = mouse_region[frame_idx]
        region_other = other_region[frame_idx]
        
        cv2.putText(frame, f"Rat{mouseID}: {region_self}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Rat{1 - mouseID}: {region_other}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        #Dimensions – 1392 x 640
        
        #Lever Gazing
        levBL = (100, 240) #Top
        levTR = (160, 180)
        cv2.rectangle(frame, levBL, levTR, (50, 50, 200), 3)
        
        levBL2 = (100, 415) #Bottom
        levTR2 = (160, 475)
        cv2.rectangle(frame, levBL2, levTR2, (0, 0, 255), 3)
        
        #Mag Gazing
        magBL = (1240, 245) #Top
        magTR = (1300, 185)
        cv2.rectangle(frame, magBL, magTR, (50, 50, 200), 3)
        
        magBL2 = (1240, 415) #Bottom
        magTR2 = (1300, 475)
        cv2.rectangle(frame, magBL2, magTR2, (0, 0, 255), 3)
        
        
        #Draw Sub-Zones: 
        zone_color = (255, 255, 255)  # White color for rectangles
        thickness = 2
        
        # Convert corner definitions to top-left and bottom-right for OpenCV
        zones = [
            ("levTop", loader.levTopTR, loader.levTopBL),
            ("levBot", loader.levBotTR, loader.levBotBL),
            ("magTop", loader.magTopTR, loader.magTopBL),
            ("magBot", loader.magBotTR, loader.magBotBL)
        ]
        
        for label, tr, bl in zones:
            top_left = (bl[0], tr[1])     # x from BL, y from TR
            bottom_right = (tr[0], bl[1]) # x from TR, y from BL
            cv2.rectangle(frame, top_left, bottom_right, zone_color, thickness)
            cv2.putText(frame, label, (top_left[0] + 5, top_left[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)
        
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
            
        # Compute and display state
        if np.any(np.isnan(pos_data[frame_idx])):
            state = 6  # exploring (NaN case)
        else:
            x, y = pos_data[frame_idx]
            vel = velocities[frame_idx]
            if (frame_idx>2):
                vel_before = np.mean(velocities[frame_idx - 2:frame_idx])
            else:
                vel_before = 0
            
        
            cv2.putText(frame, f"Vel: {vel}", (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if frame_idx in press_frames:
                state = 4  # pressed
            elif frame_idx in reward_frames:
                state = 5  # reward taken
            elif frame_idx in false_mag_entry:
                state = 7 #mag entered but no reward
            elif lever_zone[frame_idx]:
                state = 3  # waiting
            elif vel > 8 and loader.approachingMagazine(mouseID, frame_idx):
                state = 2  # approaching reward
            elif vel < 10 and vel_before < 10:
                state = 0  # idle
            elif vel > 8 and loader.approachingLever(mouseID, frame_idx):
                state = 1  # approaching lever
            else:
                state = 6  # exploring
        cv2.putText(frame, f"State: {state_names[state]}", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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
    
 
    


def visualize_gaze_overlay2(
    video_path,
    loader,
    lev,
    mag,
    mouseID=0,
    save_path="output.mp4",
    start_frame=0,
    max_frames=1000,
    gaze_length=400
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
    
    # State-related data
    state_names = ["idle", "approaching lever", "approaching reward", "waiting", "pressed", "reward taken", "exploring", "false mag"]
    pos_data = loader.getHeadBodyTrajectory(mouseID).T  # Shape: (num_frames, 2)
    velocities = loader.computeVelocity(mouseID)
    lever_zone = loader.getLeverZone(mouseID)
    reward_zone = loader.getRewardZone(mouseID)
    press_frames = lev.getLeverPressFrames(mouseID)
    reward_frames = mag.getRewardReceivedFrames(mouseID)
    false_mag_entry = mag.getEnteredMagFrames(mouseID)
    distances = loader.returnInteractionDistance()
    print("Reward_Frames: ", reward_frames)
    
    self_intersecting_rat1 = loader.checkSelfIntersection(1)
    #print("self_intersecting_rat1: ", self_intersecting_rat1)
    
    frame_idx = start_frame
    frame_count = 0
    
    while cap.isOpened() and frame_idx < num_frames and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret or frame_idx >= loader.data.shape[-1]:
            break

        gaze_vec = gaze_vector[:, frame_idx]
        gaze_origin = HB[:, frame_idx]
        target = other_body[:, :, frame_idx]
        #cv2.putText(frame, f"InterMouseDistance: {distances[frame_idx]}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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
        
        # Draw zone boundaries
        cv2.line(frame, (loader.levBoundary, 0), (loader.levBoundary, height), (255, 255, 0), 2)   # Cyan line for levBoundary
        cv2.line(frame, (loader.magBoundary, 0), (loader.magBoundary, height), (255, 0, 255), 2)   # Magenta line for magBoundary

        # Optional: label the zones
        #cv2.putText(frame, "Lev", (loader.levBoundary//2 - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        cv2.putText(frame, "Mid", (width//2 - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
        #cv2.putText(frame, "Mag", (loader.magBoundary + (width - loader.magBoundary)//2 - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)        
        
        #Mouse Region
        region_self = mouse_region[frame_idx]
        region_other = other_region[frame_idx]
        
        #cv2.putText(frame, f"Rat{mouseID}: {region_self}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        #cv2.putText(frame, f"Rat{1 - mouseID}: {region_other}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        #Draw Sub-Zones: 
        zone_color = (255, 255, 255)  # White color for rectangles
        thickness = 2
        
        # Convert corner definitions to top-left and bottom-right for OpenCV
        zones = [
            ("Top Lever", loader.levTopTR, loader.levTopBL),
            ("Bottom Lever", loader.levBotTR, loader.levBotBL),
            ("Top Mag", loader.magTopTR, loader.magTopBL),
            ("Bottom Mag", loader.magBotTR, loader.magBotBL)
        ]
        
        for label, tr, bl in zones:
            top_left = (bl[0], tr[1])     # x from BL, y from TR
            bottom_right = (tr[0], bl[1]) # x from TR, y from BL
        
            # Draw zone rectangle
            cv2.rectangle(frame, top_left, bottom_right, zone_color, thickness)
        
            # === Add text with background ===
            text_x = top_left[0] + 5
            text_y = top_left[1] + 25
        
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
            # Define rectangle for background (with some padding)
            rect_top_left = (text_x - 2, text_y - text_height - 4)
            rect_bottom_right = (text_x + text_width + 2, text_y + 4)
        
            # Draw light gray rectangle background
            cv2.rectangle(frame, rect_top_left, rect_bottom_right, (200, 200, 200), thickness=-1)
        
            # Draw the text over the background
            cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)
        
        
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
            
        
        # Compute and display state
        if np.any(np.isnan(pos_data[frame_idx])):
            state = 6  # exploring (NaN case)
        else:
            x, y = pos_data[frame_idx]
            vel = velocities[frame_idx]
            if (frame_idx>2):
                vel_before = np.mean(velocities[frame_idx - 2:frame_idx])
            else:
                vel_before = 0
            
        
            #cv2.putText(frame, f"Vel: {vel}", (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
    
 
h5_file = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/4_nanerror_test.h5"
lev_file = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/4_nanerror_lev.csv"
mag_file = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/4_nanerror_mag.csv"
video_file = "/Users/david/Downloads/4%_nan_test.mp4"    

#h5_file = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/18_nanerror_test.h5"
#video_file = "/Users/david/Downloads/18%_nan_test.mp4"

#h5_file = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5"
#video_file = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.mp4"    


loader = posLoader(h5_file)
lev = levLoader(lev_file)
mag = magLoader(mag_file)
loader.plot_rat_heatmap()

#visualize_gaze_overlay(video_file, loader, lev, mag, mouseID=0, save_path = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Graphs/Videos/testingLevandMagGazing.mp4")

  
    
    