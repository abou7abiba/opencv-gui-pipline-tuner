"""
The Utilities for different processors
"""
import logging
import numpy as np
import cv2
import os

from pipetuner.processors.imgprocessor import ImageProcessor

# create logger
logger = logging.getLogger(__name__)

###################################
#       SmoothImage
###################################
class SmoothImage (ImageProcessor):
    def __init__(self, name, image=None, on_image_change=None, average_filter_size=3, gaussian_filter_size=1, median_filter_size=1, bilateral_filter_size=1):
        
        super().__init__(name, image, on_image_change)

        self.setParameter('average_filter_size', average_filter_size)
        self.setParameter('gaussian_filter_size', gaussian_filter_size)
        self.setParameter('median_filter_size', median_filter_size)
        self.setParameter('bilateral_filter_size', bilateral_filter_size)

        def onchangeAvgFltrSz(pos):
            filter_size = pos
            filter_size += (filter_size + 1) % 2        # make sure the filter size is odd
            self.setParameter('average_filter_size', filter_size)
            self.refresh()

        def onchangeGaussFltrSz(pos):
            filter_size = pos
            filter_size += (filter_size + 1) % 2      # make sure the filter size is odd
            self.setParameter('gaussian_filter_size', filter_size)
            self.refresh()

        def onchangeMdnFltrSz(pos):
            filter_size = pos
            filter_size += (filter_size + 1) % 2          # make sure the filter size is odd
            self.setParameter('median_filter_size', filter_size)
            self.refresh()

        def onchangeBltrlFltrSz(pos):
            filter_size = pos
            filter_size += (filter_size + 1) % 2    # make sure the filter size is odd
            self.setParameter('bilateral_filter_size', filter_size)
            self.refresh()
        
        self.setWinControl(self._name + " " + self._win_Ctrl)
        
        cv2.createTrackbar('average_filter_size', self._win_Ctrl, self.getParameter('average_filter_size'), 20, onchangeAvgFltrSz)
        cv2.createTrackbar('gaussian_filter_size', self._win_Ctrl, self.getParameter('gaussian_filter_size'), 20, onchangeGaussFltrSz)
        cv2.createTrackbar('median_filter_size', self._win_Ctrl, self.getParameter('median_filter_size'), 20, onchangeMdnFltrSz)
        cv2.createTrackbar('bilateral_filter_size', self._win_Ctrl, self.getParameter('bilateral_filter_size'), 20, onchangeBltrlFltrSz)
        
    def averageFilterSize(self):
        return self.getParameter('average_filter_size')

    def gaussianFilterSize(self):
        return self.getParameter('gaussian_filter_size')

    def medianFilterSize(self):
        return self.getParameter('median_filter_size')

    def bilateralFilterSize(self):
        return self.getParameter('bilateral_filter_size')
        
    def _render(self):
        # Convert it to Grayscale
        self.setWindow(self._name)

        self._blurred_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self._blurred_image = cv2.blur(self._blurred_image, (self.averageFilterSize(), self.averageFilterSize()))
        self._blurred_image = cv2.GaussianBlur(self._blurred_image, (self.gaussianFilterSize(), self.gaussianFilterSize()), 0)
        self._blurred_image = cv2.medianBlur(self._blurred_image, self.medianFilterSize())
        self._processed_image = cv2.bilateralFilter(self._blurred_image, 5, self.bilateralFilterSize(), self.medianFilterSize())
            
        cv2.imshow(self._name, self._processed_image)

###################################
#       EdgeFinder
###################################
class EdgeFinder (ImageProcessor):
    def __init__(self, name, image=None, on_image_change=None, min_threshold=71, max_threshold=163):
        super().__init__(name, image, on_image_change)
        super().__init__(name, image, on_image_change)

        self.setParameter('min_threshold', min_threshold)
        self.setParameter('max_threshold', max_threshold)

        def onchangeThreshold1(pos):
            self.setParameter('min_threshold', pos)
            self.refresh()

        def onchangeThreshold2(pos):
            self.setParameter('max_threshold', pos)
            self.refresh()

        self.setWinControl(self._name + " " + self._win_Ctrl)

        cv2.createTrackbar('min_threshold', self._win_Ctrl, min_threshold, 255, onchangeThreshold1)
        cv2.createTrackbar('max_threshold', self._win_Ctrl, max_threshold, 255, onchangeThreshold2)

    def min_threshold(self):
        return self.getParameter('min_threshold')

    def max_threshold(self):
        return self.getParameter('max_threshold')

    def _render(self):
        self.setWindow(self._name)

        self._processed_image = cv2.Canny(self.image, self.min_threshold(), self.max_threshold())
        cv2.imshow(self._name, self._processed_image)

###################################
#       RegionMask
###################################
class RegionMask (ImageProcessor):
    def __init__(self, name, image=None, on_image_change=None, x_up=0.06, x_bottom=0.42, y_up=0.4, y_bottom=0):
        super().__init__(name, image, on_image_change)

        self.setParameter('x_up', x_up)
        self.setParameter('x_bottom', x_bottom)
        self.setParameter('y_up', y_up)
        self.setParameter('y_bottom', y_bottom)

        def onChangeXUp (pos):
            self.setParameter('x_up', pos/100)
            self.refresh()

        def onChangeXBottom (pos):
            self.setParameter('x_bottom', pos/100)
            self.refresh()

        def onChangeYUp (pos):
            self.setParameter('y_up', pos/100)
            self.refresh()

        def onChangeYBottom (pos):
            self.setParameter('y_bottom', pos/100)
            self.refresh()

        self.setWinControl(self._name + " " + self._win_Ctrl)

        cv2.createTrackbar('x_up', self._win_Ctrl, int(x_up*100), 50, onChangeXUp)
        cv2.createTrackbar('x_bottom', self._win_Ctrl, int(x_bottom*100), 50, onChangeXBottom)
        cv2.createTrackbar('y_up', self._win_Ctrl, int(y_up*100), 100, onChangeYUp)
        cv2.createTrackbar('y_bottom', self._win_Ctrl, int(y_bottom*100), 50, onChangeYBottom)

    def xUp(self):
        return int(self.xUpPercent() * self._xsize)

    def xBottom(self):
        return int(self.xBottomPercent() * self._xsize)

    def yUp(self):
        return int(self.yUpPercent() * self._ysize)

    def yBottom(self):
        return int(self.yBottomPercent() * self._ysize)

    def xUpPercent(self):
        return self.getParameter('x_up')

    def xBottomPercent(self):
        return self.getParameter('x_bottom')

    def yUpPercent(self):
        return self.getParameter('y_up')

    def yBottomPercent(self):
        return self.getParameter('y_bottom')

    def imageInfo (self):
        image_info = self._image_info
        if image_info is None:  #No image info passed to it.
            image_info = {}

        image_info['y_up']=self._ysize - self.yUp()
        image_info['y_bottom']=self._ysize - self.yBottom()
        image_info['x_up']=self.xUp()
        image_info['x_bottom']=self.xBottom()

        return image_info

    def vertices(self):
        ysize = self._ysize
        xsize = self._xsize
        return np.array([[(xsize/2 - self.xBottom(), ysize-self.yBottom()), \
            (xsize/2 - self.xUp(), ysize - self.yUp()), \
            (xsize/2 + self.xUp(), ysize - self.yUp()), \
            (xsize/2 + self.xBottom(), ysize-self.yBottom())]], dtype=np.int32)

    def setConfig (self, config):
        self._config = config
        for key in config:
            cv2.setTrackbarPos(key, self._win_Ctrl, int (config[key] * 100))     

    def region_of_interest(self):
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        `vertices` should be a numpy array of integer points.
        """
        #defining a blank mask to start with
        mask = np.zeros_like(self.image)   
        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(self.image.shape) > 2:
            channel_count = self.image.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, self.vertices(), ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(self.image, mask)
        return masked_image


    def _render(self):
        self.setWindow(self._name)

        self._processed_image = self.region_of_interest()
        region_selected = np.copy(self._processed_image)
        pts = self.vertices()
        #pts = pts.reshape((-1,1,2))
        cv2.polylines(region_selected,pts,True,(255,255,0), 2, cv2.FILLED)

        cv2.imshow(self._name, region_selected)

###################################
#       HoughLines
###################################
class HoughLines (ImageProcessor):
    def __init__(self, name, image=None, on_image_change=None, buffer_len = 40, rho=1, theta=1, threshold=7, \
                            min_line_length = 4, max_line_gap = 47, line_thickness = 2):
                            
        super().__init__(name, image, on_image_change)

        self._m_list_l = np.zeros(0)  #Buffer of last left lines slop m
        self._c_list_l = np.zeros(0)  #Buffer of last left lines c constant
        self._m_list_r = np.zeros(0)  #Buffer of last right lines slop m
        self._c_list_r = np.zeros(0)  #Buffer of last right lines c constant
        self._buffer_len = buffer_len       # size of the buffers

        self.setParameter('rho', rho)
        self.setParameter('theta', theta)
        self.setParameter('threshold', threshold)
        self.setParameter('min_line_length', min_line_length)
        self.setParameter('max_line_gap', max_line_gap)
        self.setParameter('line_thickness', line_thickness)

        # The resolution of the parameter r in pixels. We use 1 pixe
        def onchangeRho(pos):
            self.setParameter('rho', pos)
            self.refresh()

        # The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
        def onchangeTheta(pos):
            self.setParameter('theta', pos)
            self.refresh()

        #  The minimum number of intersections to "detect" a line
        def onchangeThreshold(pos):
            self.setParameter('threshold', pos)
            self.refresh()

        # The minimum number of points that can form a line. Lines with less than 
        # this number of points are disregarded.
        def onchangeMinLineLength(pos):
            self.setParameter('min_line_length', pos)
            self.refresh()

        # The maximum gap between two points to be considered in the same line.
        def onchangeMaxLineLength(pos):
            self.setParameter('max_line_gap', pos)
            self.refresh()

        # The maximum gap between two points to be considered in the same line.
        def onchangeLineThickness(pos):
            self.setParameter('line_thickness', pos)
            self.refresh()

        self.setWinControl(self._name + " " + self._win_Ctrl)

        cv2.createTrackbar('rho', self._win_Ctrl, rho, 10, onchangeRho)
        cv2.createTrackbar('theta', self._win_Ctrl, theta, 45, onchangeTheta)
        cv2.createTrackbar('threshold', self._win_Ctrl, threshold, 50, onchangeThreshold)
        cv2.createTrackbar('min_line_length', self._win_Ctrl, min_line_length, 100, onchangeMinLineLength)
        cv2.createTrackbar('max_line_gap', self._win_Ctrl, max_line_gap, 100, onchangeMaxLineLength)
        cv2.createTrackbar('line_thickness', self._win_Ctrl, line_thickness, 10, onchangeLineThickness)

    # The resolution of the parameter r in pixels. We use 1 pixe
    def rho(self):
        return self.getParameter('rho')

    # The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
    def thetaDegrees(self):
        return self.getParameter('theta')

    def theta(self):
        return self.thetaDegrees() * np.pi/180

    #  The minimum number of intersections to "detect" a line
    def threshold(self):
        return self.getParameter('threshold')

    # The minimum number of points that can form a line. Lines with less than 
    # this number of points are disregarded.
    def minLineLength(self):
        return self.getParameter('min_line_length')

    # The maximum gap between two points to be considered in the same line.
    def maxLineGap(self):
        return self.getParameter('max_line_gap')

    def lineThickness (self):
        return self.getParameter('line_thickness')

    """
    Got the color of the line based on the slop of this line
    This is used for debugging purpose.
    """
    def getlineColor (self, x1,y1,x2,y2):
        pos_big_color = [255, 255, 0]
        pos_les_color = [255, 255, 255]
        neg_big_color = [0, 0, 255]
        neg_les_color = [0, 255, 0]
        color = pos_les_color

        slop = (y2-y1)/(x2-x1)
        if slop >= 0 and abs (slop) >= 1:
            color = pos_big_color
        elif slop >= 0 and abs (slop) < 1:
            color = pos_les_color
        elif slop < 0 and abs (slop) >= 1:
            color = neg_big_color
        elif slop < 0 and abs (slop) < 1:
            color = neg_les_color
        
        return color

    def draw_lines(self, img, lines, color=[255, 255, 0], thickness=2):
        """       
        This function draws `lines` with `color` and `thickness`.    
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """
        if lines is None:
            return

        for line in lines:
            for x1,y1,x2,y2 in line:
                try:
                    if thickness > 0:
                        # cv2.line(img, (x1, y1), (x2, y2), self.getlineColor(x1,y1,x2,y2), thickness)
                        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
                except:
                    logger.debug ("Draw line failed, x1, y1: %s, %s - x2, y2: %s, %s", x1, y1, x2, y2)

    def get_line (self, lines, left_line = True):
        """
        NOTE: this is the function you might want to use as a starting point once you want to 
        average/extrapolate the line segments you detect to map out the full
        extent of the lane (going from the result shown in raw-lines-example.mp4
        to that shown in P1_example.mp4).  
        
        Think about things like separating line segments by their 
        slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
        line vs. the right line.  Then, you can average the position of each of 
        the lines and extrapolate to the top and bottom of the lane.
        
        This function draws `lines` with `color` and `thickness`.    
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """
        x_points = [] # x_points[0] will present the left line while x_points[1] is the right line
        y_points = [] # y_points[0] will present the left line while y_points[1] is the right line
        x_points.append([])
        y_points.append([])
        
        ysize = self._ysize
        xsize = self._xsize

        if lines is not None:
            for line in lines:
                for x1,y1,x2,y2 in line:
                    try:
                        slop = (y2-y1)/(x2-x1)
                        if left_line and slop < 0 and x1 < xsize/2 and x2 < xsize/2: # -ve slop is left line and where y > x
                            x_points[0].append (x1)
                            x_points[0].append (x2)
                            y_points[0].append (y1)
                            y_points[0].append (y2)
                        elif not left_line and slop >= 0 and x1 >= xsize/2 and x2 >= xsize/2: # +ve slop is right line and where y > x
                            x_points[0].append (x1)
                            x_points[0].append (x2)
                            y_points[0].append (y1)
                            y_points[0].append (y2)
                    except:
                        logger.debug ("invalid slop, x1, y1: %s, %s - x2, y2: %s, %s", x1, y1, x2, y2)
        
        # Identify left line
        line = None

        if left_line:
            m_list = self._m_list_l
            c_list = self._c_list_l
        else:
            m_list = self._m_list_r
            c_list = self._c_list_r


        y_bottom = self._image_info['y_bottom']
        y_up = self._image_info['y_up']

        x_bottom = self._image_info['x_bottom']
        x_up = self._image_info['x_up']
        x_bottom_l, x_bottom_r = xsize/2 - x_bottom, xsize/2 + x_bottom
        x_up_l, x_up_r = xsize/2 - x_up, xsize/2 + x_up

        x = np.array(x_points[0])
        y = np.array(y_points[0])

        # y_bottom = ysize -1
        # y_up = min(y)

        m, c = None, None

        if len(x) > 0 and len (y) > 0:
            line_fit = np.polyfit(x, y, 1)
        
            # y = mx + c 
            # x = (y - c)/m 
            # So we get x at the point in the image = ysize -1 to draw complete lines
            m, c = line_fit

            in_range = False
            x1 = int((y_bottom -c)/m)
            x2 = int((y_up -c)/m)
            
            if x1 >= x_bottom_l and x1 <= x_bottom_r and x2 >= x_up_l and x2 <= x_up_r:
                in_range = True

            if in_range:    
                # Save both m an c in a buffer
                m_list = np.insert(m_list, 0, m)
                c_list = np.insert(c_list, 0, c)

                if len(m_list) > self._buffer_len:
                    m_list = np.delete(m_list, -1)
                    c_list = np.delete(c_list, -1)
                
                # get the avarage of the list as the new value of m & c
                m = np.mean(m_list)
                c = np.mean(c_list)

                if left_line: # Keep the updated arrays 
                    self._m_list_l = m_list
                    self._c_list_l = c_list
                else:
                    self._m_list_r = m_list
                    self._c_list_r = c_list

            elif len(m_list) > 0: # ignore the values of this line and use the last values
                m = np.mean(m_list)
                c = np.mean(c_list)

            if m is not None:
                line = np.array([int((y_bottom -c)/m), y_bottom, int((y_up -c)/m), y_up])
        
        elif len(m_list) > 0:
            m = np.mean(m_list)
            c = np.mean(c_list)
            line = np.array([int((y_bottom -c)/m), y_bottom, int((y_up -c)/m), y_up])
        
        if line is None:
            logger.debug ("No line for input lines: %s and left_line is: %s", lines, left_line)

        return line
   
    
    def draw_lane(self, img, lines, color=[255, 0, 0]):
        """
        NOTE: this is the function you might want to use as a starting point once you want to 
        average/extrapolate the line segments you detect to map out the full
        extent of the lane (going from the result shown in raw-lines-example.mp4
        to that shown in P1_example.mp4).  
        
        Think about things like separating line segments by their 
        slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
        line vs. the right line.  Then, you can average the position of each of 
        the lines and extrapolate to the top and bottom of the lane.
        
        This function draws `lines` with `color` and `thickness`.    
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """
        lane_lines = np.empty([0, 1, 4], dtype=int)
        left_line = self.get_line (lines, left_line = True)

        if left_line is not None:
            lane_lines = np.append (lane_lines, [[left_line]], axis=0)

        right_line = self.get_line (lines, left_line = False)
        if right_line is not None:
            lane_lines = np.append (lane_lines, [[right_line]], axis=0)

         
        #logger.debug('lane lines are : %s', lane_lines)
        
        self.draw_lines (img, lane_lines, color, self.lineThickness())

    def _render(self):
        self.setWindow(self._name)

        # Make a blank the same size as our image to draw on
        line_imge = np.zeros((self._ysize, self._xsize, 3), dtype=np.uint8)

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(self.image, self.rho(), self.theta(), self.threshold(), np.array([]),
                                    self.minLineLength(), self.maxLineGap())

        self.draw_lines(line_imge, lines)
        self.draw_lane (line_imge, lines)

        #Fix the colors
        r, g, b = cv2.split (line_imge)
        fixed_image = cv2.merge ((b, g, r))

        self._processed_image = fixed_image
        cv2.imshow(self._name, self._processed_image)

###################################
#       ImageBlender
###################################
class ImageBlender (ImageProcessor):
    def __init__(self, name, image=None, base_image=None, on_image_change=None, alpha=0.8, beta=1, gamma=0):
        super().__init__(name, image, on_image_change)

        self.setParameter('alpha', alpha)
        self.setParameter('beta', beta)
        self.setParameter('gamma', gamma)

        self._base_image = base_image

        def onchangeAlpha(pos):
            self.setParameter('alpha', pos / 10)
            self.refresh()

        def onchangeBeta(pos):
            self.setParameter('beta', pos / 10)
            self.refresh()

        def onchangeGamma(pos):
            self.setParameter('gamma', pos / 10)
            self.refresh()

        self.setWinControl(self._name + " " + self._win_Ctrl)
 
        cv2.createTrackbar('alpha', self._win_Ctrl, int(alpha * 10), 10, onchangeAlpha)
        cv2.createTrackbar('beta', self._win_Ctrl, int(beta * 10), 10, onchangeBeta)
        cv2.createTrackbar('gamma', self._win_Ctrl, int(gamma * 10), 10, onchangeGamma)

    def alpha(self):
        return self.getParameter('alpha')
    
    def beta(self):
        return self.getParameter('beta')

    def gamma(self):
        return self.getParameter('gamma')

    def setBaseImage (self, base_image):
        self._base_image = base_image

    def setConfig (self, config):
        self._config = config
        for key in config:
            cv2.setTrackbarPos(key, self._win_Ctrl, int (config[key] * 10))     

    def getImageInfo (self, key):
        if self._image_info is not None and key in self._image_info:
            return str(self._image_info[key])
        else:
            return "NA"

    def showImageInfo (self, image):
        # describe the type of font 
        # to be used. 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        
        file = self.getImageInfo('file')
        frame = self.getImageInfo('frame')
        frames_num = self.getImageInfo('frames_num')
        wait_time = self.getImageInfo('wait_time')

        image_info = {'file': file, 'frame': frame + "/" + frames_num, 'wait time': wait_time}
        
        i, y0, dy = 0, 50, 30
        if image_info is not None:
            for key, value in image_info.items():
                y = y0 + i*dy
                i+=1
                # Use putText() method for 
                # inserting text on video 
                cv2.putText(image,  
                            str(key) + ": " + str(value),  
                            (50, y),  
                            font, 1,  
                            (255, 255, 255),  
                            2,  
                            cv2.LINE_8) 

    def _render(self):
        base_image = self._pipline.getImage()
        self.setWindow(self._name)

        self.showImageInfo (base_image)
        self._processed_image = cv2.addWeighted(base_image, self.alpha(), self.image, self.beta(), self.gamma())
        cv2.imshow(self._name, self._processed_image)

