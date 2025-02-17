import numpy as np
import cv2
import pypupilext as pp
import matplotlib.pyplot as plt
import os

debug = True #generates lists of options, ignores given parameters
verbose = True #shows everything generated

class Pupil:
    def __init__(self, xloc, yloc, diameter, maxHist = 10):
        """
        Initialise an instance of the pupil class
        Args:
            xloc (int): the x location of the pupil
            yloc (int): the y location of the pupil
            diameter (int): the diameter of the pupil
        """
        self.x = xloc
        self.y = yloc
        self.d = diameter
        self.xHist = []
        self.yHist = []
        self.dHist = []
        self.maxHist = maxHist
        return self
    
    def set_x(self, x):
        self.xHist.append(self.x)
        if len(self.xHist) > self.maxHist:
            del self.xHist[0]
        self.x = x
        return self
    
    def set_y(self, y):
        self.yHist.append(self.y)
        if len(self.yHist) > self.maxHist:
            del self.yHist[0]
        self.y = y
        return self
    
    def set_d(self, d):
        self.dHist.append(self.d)
        if len(self.dHist) > self.maxHist:
            del self.dHist[0]
        self.d = d
        return self
    
    def get_diameter(self):
        return self.d
    
    def get_y(self):
        return self.y
    
    def get_x(self):
        return self.x
    
    def get_yHist(self):
        return self.yHist
    
    def get_xHist(self):
        return self.xHist
    
    def get_dHist(self):
        return self.dHist

class Tracker:
    def __init__(self, pupilModel="Starburst", faceModelPath='data/haarcascades/haarcascade_frontalface_default.xml', eyeModelPath='data/haarcascades/haarcascade_eye.xml'):
        """
        Initialise tracker

        Args:
            pupilModel (str="Starburst"): the pupil detection/tracking method to be used. 
                accepted values: {'ElSe', 'ExCuSe', 'PuRe', 'PuReST', 'Starburst', Swirski2D'} 
                    OR a str(dict) of the format "{'pre': [{'none', 'blur', 'threshLow', 'threshMid', 
                                                    'threshHigh', 'erosion', 'dilation', 'opening', 
                                                    'closing', 'morphGradient', 'blackHat', 'topHat', 
                                                    'sobelx', 'sobely', 'sobelxy', 'canny', 'laplacian'}],
                                            'post': {'blob', 'hough', 
                                                    'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 
                                                    'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 
                                                    'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'}}"
            faceModelPath (str): the filepath for the haar cascade face classifier
            eyeModelPath (str): the filepath for the haar cascade eye classifier
        Returns:
            None
        """
        if debug:
            pupilModel = "{'pre': 'all', 'post': 'blob'}"
        self.faceModel = cv2.CascadeClassifier(faceModelPath)
        self.eyeModel = cv2.CascadeClassifier(eyeModelPath)
        match pupilModel:
            case "ElSe":
                self.pupilModel = pp.ElSe()
                self.pupilExtModel = True
            case "ExCuSe":
                self.pupilModel = pp.ExCuSe()
                self.pupilExtModel = True
            case "PuRe":
                self.pupilModel = pp.PuRe()
                self.pupilExtModel = True
            case "PuReST":
                self.pupilModel = pp.PuReST()
                self.pupilExtModel = True
            case "Starburst":
                self.pupilModel = pp.Starburst()
                self.pupilExtModel = True
            case "Swirski2D":
                self.pupilModel = pp.Swirski2D()
                self.pupilExtModel = True
            case _:
                self.pupilModelParams = eval(pupilModel)
                # TODO check params ^ are valid
                self.pupilExtModel = False
                if self.pupilModelParams["post"] == "blob":
                    #TODO these hyperparameters WILL have to be tuned
                    #TODO look at thresholding, grouping, merging, etc.
                    params = cv2.SimpleBlobDetector_Params()
                    params.filterByArea = True
                    params.maxArea = 1500
                    params.filterByCircularity = True #shape circularity
                    params.minCircularity = 0.5
                    params.filterByInertia = True #shape longness
                    params.minInertiaRatio = 0.6
                    self.pupilModel = cv2.SimpleBlobDetector(params)

    def draw_all_rectangles(self, cv2Image, objects, maxObject = [-1, -1, -1, -1], offset = (0, 0)): #TODO REDO USAGE
        """
        DEBUG METHOD - draws rectangles on an image

        Args:
            cv2Image (np.array): the image to be visualised. Entries should be in uint8 format
            objects ([int[4]]): the objects to visualise. Each object takes format [x, y, w, h]
            maxObject (int[4] = [-1,-1,-1,-1]): the object to draw in a different colour
            offset ((int, int) = (0, 0)): the offset to give the rectangles. 
                Used if searching for objects in a subsection of the original image 
        
        Returns:
            colourImg (np.array, dtype=uint8): the image with rectangles drawn on

        """
        x, y, w, h = maxObject
        xOff = offset[0]
        yOff = offset[1]
        colourImg = cv2.cvtColor(cv2Image, cv2.COLOR_GRAY2RGB)
        for object in objects:
            xLoc, yLoc, wLoc, hLoc = object
            if (xLoc, yLoc, wLoc, hLoc) == (x, y, w, h):
                c = (255, 0, 0)
            else:
                c = (0, 165, 255)
            cv2.rectangle(colourImg,  (xLoc+xOff-10, yLoc+yOff-20),
                        (xLoc+xOff + wLoc+10 , yLoc+yOff + hLoc+20),
                        c,1)
        return colourImg

    def draw_pupil(self, cv2Image, pupil, offset=(0,0)):
        """
        DEBUG METHOD - draws a circle on an image

        Args:
            cv2Image (np.array): the image to be visualised. Entries should be in uint8 format
            pupil (x(int), y(int), d(int)): the pupil to draw - (x, y) of center, and diameter
            offset ((int, int) = (0, 0)): the offset to give the rectangles. 
                Used if searching for objects in a subsection of the original image 
        
        Returns:
            colourImg (np.array, dtype=uint8): the image with a circle drawn on

        """
        x = pupil[0]
        y = pupil[1]
        d = pupil[2]
        # colourImg = cv2.cvtColor(cv2Image, cv2.COLOR_GRAY2RGB)
        colourImg = cv2Image.copy()
        #draw circumference
        cv2.circle(colourImg, (x, y), d//2, (255, 0, 0), 1)
        #draw center
        cv2.circle(colourImg, (x, y), 1, (0, 255, 0), 1) 
        return colourImg

    def show(self, cv2Image, label="Image", note=None, scale=1, destroy=False): #TODO replace all cv2.imshow with this
        """
        DEBUG METHOD to show an image in the openCV window

        Args:
            cv2Image (np.array. dtype=uint8): the image to display
            label (str="Image"): the label for the window
            note (str=None): the note to print when displaying the image, if any
            scale (float=1): used to resize the image in all dimensions
            destroy (bool=True): whether to destroy all windows

        Returns:
            None
        """
        if note != None:
            print(note)
        plt.imshow(cv2Image)
        plt.show()
        # resize = cv2.resize(cv2Image, None, fx=scale, fy=scale)
        # cv2.imshow(label, resize)
        # cv2.waitKey(0)
        # if destroy:
        #     cv2.destroyAllWindows()

    def find_face(self, cv2Image):
        """
        Find faces in an image using the OpenCV haar cascade model. 
        Returns the original image cropped to just show the largest face and the coordinates of that face
        If no face is detected, return the original image and (0, 0)

        Args:
            cv2Image (np.array, dtype=uint8): the image to detect faces in 
        
        Returns:
            cv2Image (np.array, dtype=uint8): the original image, cropped to just show the detected face
            (x, y) (int, int): the x and y coordinates of the detected face
        """
        faces = self.faceModel.detectMultiScale(cv2Image, 1.3, 5)
        if len(faces) == 0:
            return cv2Image, (0, 0)
        maxFace= faces[np.argmax(faces, 0)[2]].tolist() #get biggest detected face, measured by width
        x, y, w, h = maxFace
        if debug:
            if verbose:
                rects = self.draw_all_rectangles(cv2Image, faces, maxFace)
                self.show(rects, label="faces detected")
        return cv2Image[y:y+h, x:x+w], (x, y)
    
    def find_eyes(self, cv2Image, xFace=0, yFace=0, wFace=-1, hFace=-1):
        """
        Find eyes in an image using the OpenCV haar cascade model.
        Returns the original image cropped to just show the rightmost eye, and the eye's coordinates in the image
        If no eyes are detected, return the original image and (0, 0)

        Args:
            cv2Image (np.array, dtype=uint8): the image to detect eyes in 
            xFace (int=0): the x-coordinate of the detected eye
            yFace (int=0): the y-coordinate of the detected eye
            wFace (int=-1): the width of the detected eye
            hFace (int=-1): the height of the detected eye
        
        Returns:
            cv2Image (np.array, dtype=uint8): the original image, cropped to just show the detected eye
            (x, y) (int, int): the x and y coordinates of the detected eye
        """
        #TODO simple data assumptions? eyes will only be in the top half of the model, for e.g. - simplify computation
        if wFace == -1:
            wFace, hFace = cv2Image.shape #TODO RIGHT DIMENSIONS??
        eyes = self.eyeModel.detectMultiScale(cv2Image[yFace:yFace+hFace, xFace:xFace+wFace], 1.3, 5)
        if len(eyes) == 0:
            return cv2Image, (0, 0)
        maxEye = eyes[np.argmax(eyes, 0)].tolist()[0] 
        x, y, w, h  = maxEye
        if verbose:
            rects = self.draw_all_rectangles(cv2Image, eyes, maxEye)
            self.show(rects, label="eyes detected")
        return cv2Image[y:y+h, x:x+w], (x, y)

    def preprocess_opencv(self, cv2Image, method):
        """
        preprocessing for data-based pupil detection.
        
        Args:
            cv2Image (np.array, dtype=uint8): the image to be preprocessed
            method (str): the method to preprocess. Accepted values:
                {"none", "blur", "threshLow", "threshMid", "threshHigh", 
                "erosion", "dilation", "opening", "closing", "morphGradient", 
                "blackHat", "topHat", "sobelx", "sobely", "sobelxy", "canny", 
                "laplacian"}
        
        Returns:
            img (np.array, dtype=uint8): the processed image
        """
        match method:
        #gaussian blur
            case "blur":
                img = cv2.GaussianBlur(cv2Image, (3, 3), sigmaX=0, sigmaY=0)
        #binarizing / thresholding
            case "threshLow":
                _, img = cv2.threshold(cv2Image, 45, 255, cv2.THRESH_BINARY)
            case "threshMid":
                _, img = cv2.threshold(cv2Image, 127, 255, cv2.THRESH_BINARY)
            case "threshHigh":
                _, img = cv2.threshold(cv2Image, 200, 255, cv2.THRESH_BINARY)
        # morphological changes
            case "erosion":
                binr = cv2.threshold(cv2Image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] 
                kernel = np.ones((5, 5), np.uint8) 
                invert = cv2.bitwise_not(binr) 
                img = cv2.erode(invert, kernel, iterations=1) 
            case "dilation":
                binr = cv2.threshold(cv2Image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] 
                kernel = np.ones((3, 3), np.uint8) 
                invert = cv2.bitwise_not(binr) 
                img = cv2.dilate(invert, kernel, iterations=1) 
            case "opening":
                binr = cv2.threshold(cv2Image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] 
                kernel = np.ones((3, 3), np.uint8) 
                img = cv2.morphologyEx(binr, cv2.MORPH_OPEN, kernel, iterations=1) 
            case "closing":
                binr = cv2.threshold(cv2Image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] 
                kernel = np.ones((3, 3), np.uint8) 
                img = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, kernel, iterations=1) 
            case "morphGradient":
                binr = cv2.threshold(cv2Image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] 
                kernel = np.ones((3, 3), np.uint8) 
                invert = cv2.bitwise_not(binr) 
                img = cv2.morphologyEx(invert, cv2.MORPH_GRADIENT,  kernel) 
            case "blackHat":
                binr = cv2.threshold(cv2Image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] 
                kernel = np.ones((5, 5), np.uint8) 
                invert = cv2.bitwise_not(binr) 
                img = cv2.morphologyEx(invert, cv2.MORPH_BLACKHAT, kernel) 
            case "topHat":
                binr = cv2.threshold(cv2Image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] 
                kernel = np.ones((13, 13), np.uint8) 
                invert = cv2.bitwise_not(binr) 
                img = cv2.morphologyEx(invert, cv2.MORPH_TOPHAT, kernel) 
        # Edge Detection
            case "sobelx":
                img = cv2.Sobel(src=cv2Image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
            case "sobely":
                img = cv2.Sobel(src=cv2Image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
            case "sobelxy":
                img = cv2.Sobel(src=cv2Image, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
            case "canny":
                img = cv2.Canny(image=cv2Image, threshold1=100, threshold2=200) 
            case "laplacian":
                img = cv2.Laplacian(cv2Image,cv2.CV_64F)
            case _:
                if method != "none":
                    print(f"BAD METHOD: {method}")
                return cv2Image.copy()
        #rescale image for use in further processing steps
        img = np.uint8(cv2.normalize(img, np.zeros(img.shape), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
        return img

    def get_preprocessed_list(self, cv2Image, multiLevel=False):
        """
        DEBUG METHOD to generate a list of all possible preprocessed images, to test the effectiveness
        of different method combinations.

        Args:
            cv2Image (np.array, dtype=uint8): the image to process
            multiLevel (bool=False): whether to apply one stage of preprocessing, or two stages.
        
        Returns
            methodOutputs ([np.array, dtype=uint8]): the list of all preprocessed images
            methodLabels ([str]): labels for each image in methodOutputs
        """
        methods = ["blur", "threshLow", "threshMid", "threshHigh", "erosion", "dilation", "opening", "closing", "morphGradient", "blackHat", "topHat", "sobelx", "sobely", "sobelxy", "canny", "laplacian"]
        methodLabels = ["none"]
        methodOutputs = [cv2Image.copy()]
        for method1 in methods:
            img = self.preprocess_opencv(cv2Image, method1)
            methodOutputs.append(img)
            methodLabels.append(method1)
            # if multiLevel:
            #     for method2 in methods:
            #         out = self.preprocess_opencv(img.copy(), method2)
            #         methodOutputs.append(out)
            #         methodLabels.append(f"{method1}>{method2}")
        # if verbose:
        #     self.show(cv2.hconcat(methodOutputs), label="preprocessed", note=" | ".join(methodLabels))
        return methodOutputs, methodLabels
    
    def generate_template(self, diameter=7):
        """
        Generates a template for a pupil of a given diameter, for use in template matching pupil detection
        
        Args:
            diameter (int=7): the diameter of the pupil, in pixels
        
        Returns:
            grid (np.array, dtype=uint8): a black circle on a grey background (the template for the pupil)
        """
        grid = np.zeros((diameter, diameter))
        for i in range(diameter):
            for j in range(diameter):
                if np.linalg.norm(((diameter/2)-i, (diameter/2)-j)) >= diameter/2:
                    grid[i,j] = 100
        return grid.astype('uint8')

    def process_opencv(self, cv2Image, method):
        """
        Performs pupil detection using data-based methods from the openCV library
        
        Args:
            cv2Image (np.array, dtype=uint8)
            method (str): the method to use for pupil detection. Possible values:
                {'blob', 'hough', 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'}
        
        Returns:
            x (int): the x-coordinate of the pupil
            y (int): the y-coordinate of the pupil
            d (int): the diameter of the pupil
        """
        if method == "blob": #TODO
            #contour detection
            contours, hierarchy = cv2.findContours(cv2Image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(cv2Image, contours, -1, (255,0,0), 2)
            keypoints = self.pupilModel.detect(cv2Image)
            #TODO unknown C# error and probably honestly not worth fixing, it's not going to be nearly as precise as the pupilEXT stuff

        elif method == "hough":
            # Hough transform - search for circles
            detected_circles = cv2.HoughCircles(cv2Image, cv2.HOUGH_GRADIENT, 
                                                1, 20, param1 = 50, param2 = 30, 
                                                minRadius = 1, maxRadius = 40) 
            if detected_circles is not None: 
                # Convert the circle parameters a, b and r to integers. 
                detected_circles = np.uint16(np.around(detected_circles))
                if verbose:
                    for pt in detected_circles[0, :]:
                        cv2Image = self.draw_pupil(cv2Image, (pt[0], pt[1], pt[2]*2))
                    self.show(cv2Image, label="Hough Transform Detected Circles")
                pt = detected_circles[0,0] #TODO is this right
                return pt[0], pt[1], pt[2]*2
            else:
                print("No circles detected with Hough transform")
                return 0, 0, 0
        
        elif method.startswith("cv2"):
            #template matching
            template = self.generate_template(7)
            w, h = template.shape[::-1]
            method = eval(method)
            res = cv2.matchTemplate(cv2Image,template,method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            if verbose:
                rects = self.draw_all_rectangles(cv2Image, [(top_left[0], top_left[1], w, h)])
                self.show(rects, label="Template Matching")
            return top_left[0] + w//2, top_left[1] + h//2, w
        else:
            print("""
            BAD METHOD GIVEN FOR PUPIL DETECTION. Valid methods are:
            'blob', 'hough', 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'
            """)

    def get_processed_list(self, cv2Image, preStages = "processed"):
        """
        DEBUG FUNCTION that takes an image and applies all possible pupil detection methods, then displays them
        
        Args:
            cv2Image (np.array, dtype=uint8): the image to process
            preStages (str='processed'): the label of the display window

        Returns:
            pupils ([pupil]): the list of pupils, where each pupil is a tuple
                with attributes (x (int), y (int), d (int))
        """
        images = []
        pupils = []
        # NB removed 'blob' from methods because it's broken and will take *forever* to fix
        methods = ['hough', 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 
                        'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 
                        'cv2.TM_SQDIFF_NORMED']
        for method in methods:
            x, y, d = self.process_opencv(cv2Image, method)
            img = self.draw_pupil(cv2Image, (x, y, d))
            images.append(img)
            pupils.append((x, y, d))
        self.show(cv2.hconcat(images), label=preStages, note=' | '.join(methods))
        return pupils

    def find_pupil_cv2(self, cv2Image):
        """
        finds pupil in an image using openCV and (if debug==False) 
        the methods set in __init__, or (if debug==True) all method combinations
        
        Args:
            cv2Image (np.array, dtype=uint8): the image to process
        
        Returns:
            pupil (x(int), y(int), d(int)): the coordinates of the pupil found
        """
        if debug:
            pre, preLabels = self.get_preprocessed_list(cv2Image, multiLevel=False)
            for i in range(len(pre)):
                post = self.get_processed_list(pre[i], preLabels[i])
                pupil = post[-1]
        else:
            for step in self.pupilModelParams['pre']:
                cv2Image = self.preprocess_opencv(cv2Image, step)
            pupil = self.process_opencv(cv2Image, self.pupilModelParams['post'])
        return pupil
    
    def find_pupil_pupilEXT(self, cv2Image):
        """
        finds a pupil in an image using an algorithm from https://github.com/openPupil/PyPupilEXT
        
        Args:
            cv2Image (np.array, dtype=uint8): the image to find a pupil in
        
        Returns:
            pupil 
        """
        if not debug:
            pupil = self.pupilModel.run(cv2Image)
        else:
            plotted  = []
            plotLabels = []
            models = [pp.ElSe(), pp.ExCuSe(), pp.PuRe(), pp.PuReST(), pp.Starburst(), pp.Swirski2D()]
            for model in models:
                pupil = model.run(cv2Image)
                print(f"{str(model.__class__).split(['.'][-1])} | {pupil.center}, {pupil.majorAxis()}")
                #need location & diameter
                imgCpy = cv2.cvtColor(cv2Image, cv2.COLOR_GRAY2RGB)
                plot = cv2.ellipse(imgCpy,
                            (int(pupil.center[0]), int(pupil.center[1])),
                            (int(pupil.minorAxis()/2), int(pupil.majorAxis()/2)), pupil.angle,
                            0, 360, (0, 0, 255), 1)
                plotted.append(plot)
                plotLabels.append(f"{str(model.__class__).split('.')[-1][:-2]}: {np.around(pupil.center)}, {pupil.majorAxis()}")
            self.show(cv2.hconcat(plotted), "Pupils Detected", ' | '.join(plotLabels))
        return pupil

    def track(self, cv2Image, ID):
        grayscale = cv2.cvtColor(cv2Image, cv2.COLOR_BGR2GRAY)
        if self.pupilExtModel:
            pupil = self.find_pupil_pupilEXT(grayscale)
        else:
            pupil = self.find_pupil_cv2(grayscale)
        # pysource eye motion tracking opencv with python
        # medium also does it
        #TODO check how stable the eye tracker is OR find eye edges (ideally both)
        pass

def main():
    # testing.
    debug=True
    verbose=True
    # https://www.guidodiepen.nl/2017/02/detecting-and-tracking-a-face-with-python-and-opencv/
    baseImage = cv2.imread('data/images/genericFace.png')
    resultImage = baseImage.copy()
    gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
    tracker = Tracker()
    face, faceLoc = tracker.find_face(gray)
    eye, eyeLoc = tracker.find_eyes(face)
    pupil = tracker.find_pupil_pupilEXT(eye.copy())
    pupil2 = tracker.find_pupil_cv2(eye)
    print(f"{pupil.majorAxis()}, {pupil.minorAxis()}")
    

if __name__ == "__main__":
    main()

