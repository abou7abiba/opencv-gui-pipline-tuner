import numpy as np
import cv2
import os

###################################
#       ImageProcessor
###################################
class ImageProcessor:
    def __init__(self, name, image=None, on_image_change=None):
        self._name = name
        self._config = {}
        self.image = image
        self._on_image_change = on_image_change
        self._processed_image = None
        self._win_Ctrl = 'Controls'


    def __str__( self ) :
        return "( " + "name: " + str( self._name ) + ", " + str(self._config) + ")\n"

    def onImageChange(self, image_processor):
        self.image = image_processor.processedImage()
        self._ysize = self.image.shape[0]
        self._xsize = self.image.shape[1]
        self.refresh()

    def addProcessor (self, image_processor):
        self._on_image_change = image_processor

    def onCloseWindow(self):
        cv2.destroyWindow(self._name)

    def processedImage (self):
        return self._processed_image

    def name (self):
        return self._name

    def _render(self):
       pass

    def setParameter(self, name, value):
       self._config [name] = value

    def getParameter (self, name):
        return self._config [name]

    def configuration(self):
        return self._config

    def setConfig (self, config):
       self._config = config
       for key in config:
        cv2.setTrackbarPos(key, self._win_Ctrl, config[key])     

    def saveProcessedImage(self, image_filename, out_folder):
        (head, tail) = os.path.split(image_filename)
        (root, ext) = os.path.splitext(tail)

        out_filename = os.path.join(out_folder, root + '-' + self.name() + ext)
        cv2.imwrite(out_filename, self.processedImage())

    def refresh(self):
        self._render()
        # Notify the following processor in the pipline with the update 
        # to force refresh to following Processors
        if self._on_image_change and callable(self._on_image_change):
            self._on_image_change(self)

###################################
#       SmoothImage
###################################
class SmoothImage (ImageProcessor):
    def __init__(self, name, image, on_image_change=None, average_filter_size=3, gaussian_filter_size=1, median_filter_size=1, bilateral_filter_size=1):
        
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

        cv2.namedWindow(self._win_Ctrl, cv2.WINDOW_NORMAL)

        cv2.createTrackbar('average_filter_size', self._win_Ctrl, self.getParameter('average_filter_size'), 20, onchangeAvgFltrSz)
        cv2.createTrackbar('gaussian_filter_size', self._win_Ctrl, self.getParameter('gaussian_filter_size'), 20, onchangeGaussFltrSz)
        cv2.createTrackbar('median_filter_size', self._win_Ctrl, self.getParameter('median_filter_size'), 20, onchangeMdnFltrSz)
        cv2.createTrackbar('bilateral_filter_size', self._win_Ctrl, self.getParameter('bilateral_filter_size'), 20, onchangeBltrlFltrSz)
        
        #self._render()


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
        cv2.namedWindow(self._name, cv2.WINDOW_KEEPRATIO)
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
    def __init__(self, name, image, on_image_change=None, min_threshold=71, max_threshold=163):
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

        cv2.namedWindow(self._win_Ctrl, cv2.WINDOW_NORMAL)

        cv2.createTrackbar('min_threshold', self._win_Ctrl, min_threshold, 255, onchangeThreshold1)
        cv2.createTrackbar('max_threshold', self._win_Ctrl, max_threshold, 255, onchangeThreshold2)

        #self._render()

    def min_threshold(self):
        return self.getParameter('min_threshold')

    def max_threshold(self):
        return self.getParameter('max_threshold')

    def _render(self):
        cv2.namedWindow(self._name, cv2.WINDOW_KEEPRATIO)
        self._processed_image = cv2.Canny(self.image, self.min_threshold(), self.max_threshold())
        cv2.imshow(self._name, self._processed_image)

###################################
#       RegionMask
###################################
class RegionMask (ImageProcessor):
    def __init__(self, name, image, on_image_change=None, x_up=0.06, x_bottom=0.42, y_up=0.4, y_bottom=0):
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

        cv2.namedWindow(self._win_Ctrl, cv2.WINDOW_NORMAL)

        cv2.createTrackbar('x_up', self._win_Ctrl, int(x_up*100), 50, onChangeXUp)
        cv2.createTrackbar('x_bottom', self._win_Ctrl, int(x_bottom*100), 50, onChangeXBottom)
        cv2.createTrackbar('y_up', self._win_Ctrl, int(y_up*100), 100, onChangeYUp)
        cv2.createTrackbar('y_bottom', self._win_Ctrl, int(y_bottom*100), 50, onChangeYBottom)

        #self._render()


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
        cv2.namedWindow(self._name, cv2.WINDOW_KEEPRATIO)

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
    def __init__(self, name, image, on_image_change=None, rho=1, theta=1, threshold=7, \
                            min_line_length = 4, max_line_gap = 47, line_thickness = 2):
                            
        super().__init__(name, image, on_image_change)

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

        cv2.namedWindow(self._win_Ctrl, cv2.WINDOW_NORMAL)

        cv2.createTrackbar('rho', self._win_Ctrl, rho, 10, onchangeRho)
        cv2.createTrackbar('theta', self._win_Ctrl, theta, 45, onchangeTheta)
        cv2.createTrackbar('threshold', self._win_Ctrl, threshold, 50, onchangeThreshold)
        cv2.createTrackbar('min_line_length', self._win_Ctrl, min_line_length, 100, onchangeMinLineLength)
        cv2.createTrackbar('max_line_gap', self._win_Ctrl, max_line_gap, 100, onchangeMaxLineLength)
        cv2.createTrackbar('line_thickness', self._win_Ctrl, line_thickness, 10, onchangeLineThickness)

        # self._render()

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

    def draw_lines(self, img, lines, color=[255, 255, 0], thickness=2):
        """       
        This function draws `lines` with `color` and `thickness`.    
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

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
        x_points = [] # x_points[0] will present the left line while x_points[1] is the right line
        x_points.append([])
        x_points.append([])
        y_points = [] # y_points[0] will present the left line while y_points[1] is the right line
        y_points.append([])
        y_points.append([])
        
        ysize = self._ysize
        xsize = self._xsize

        for line in lines:
            for x1,y1,x2,y2 in line:
                slop = (y2-y1)/(x2-x1)
                if slop < 0 and x1 < xsize/2 and x2 < xsize/2 : # -ve slop is left line
                    x_points[0].append (x1)
                    x_points[0].append (x2)
                    y_points[0].append (y1)
                    y_points[0].append (y2)
                elif slop >= 0 and x1 >= xsize/2 and x2 >= xsize/2 : # +ve slop is right line
                    x_points[1].append (x1)
                    x_points[1].append (x2)
                    y_points[1].append (y1)
                    y_points[1].append (y2)
        
        # Identify left line
        x = np.array(x_points[0])
        y = np.array(y_points[0])
        line_fit = np.polyfit(x, y, 1)
        
        # y = mx + c 
        # x = (y - c)/m 
        # So we get x at the point in the image = ysize -1 to draw complete lines
        m, c = line_fit
        left_line = np.array([int((ysize -1 -c)/m), ysize -1, int((min(y) -c)/m), min(y)])
        
        #f = np.poly1d(line_fit)                 
        #left_line = np.array([min(x), int(f(min(x))), max(x), int(f(max(x)))])
        # logger.debug('left line segments are \n x: %s \n y: %s \n and line is \n %s', x, y, left_line)
        
        # Identify right line
        x = np.array(x_points[1])
        y = np.array(y_points[1])
        line_fit = np.polyfit(x, y, 1)
        
        m, c = line_fit
        right_line = np.array([int((ysize -1 -c)/m), ysize -1, int((min(y) -c)/m), min(y)])

        #f = np.poly1d(line_fit)                 
        #right_line = np.array ([min(x), int(f(min(x))), max(x), int(f(max(x)))])
        #logger.debug('right line segments are \n x: %s \n y: %s \n and line is \n %s', x, y, right_line)
        
        lane_lines = np.array([[left_line, right_line]]) 
        #logger.debug('lane lines are : %s', lane_lines)
        
        self.draw_lines (img, lane_lines, color, self.lineThickness())
    
    
    def _render(self):
        cv2.namedWindow(self._name, cv2.WINDOW_KEEPRATIO)

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
    def __init__(self, name, image, base_image, on_image_change=None, alpha=0.8, beta=1, gamma=0):
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

        cv2.namedWindow(self._win_Ctrl, cv2.WINDOW_NORMAL)

        cv2.createTrackbar('alpha', self._win_Ctrl, int(alpha * 10), 10, onchangeAlpha)
        cv2.createTrackbar('beta', self._win_Ctrl, int(beta * 10), 10, onchangeBeta)
        cv2.createTrackbar('gamma', self._win_Ctrl, int(gamma * 10), 10, onchangeGamma)

        # self._render()


    def alpha(self):
        return self.getParameter('alpha')
    
    def beta(self):
        return self.getParameter('beta')

    def gamma(self):
        return self.getParameter('gamma')

    def setConfig (self, config):
       self._config = config
       for key in config:
        cv2.setTrackbarPos(key, self._win_Ctrl, int (config[key] * 10))     

    def _render(self):
        cv2.namedWindow(self._name, cv2.WINDOW_KEEPRATIO)

        self._processed_image = cv2.addWeighted(self._base_image, self.alpha(), self.image, self.beta(), self.gamma())
        cv2.imshow(self._name, self._processed_image)

