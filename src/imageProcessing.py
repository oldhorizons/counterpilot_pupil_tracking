import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import skimage.exposure as exposure
# import dlib

class Pupil:
    def __init__(self, xloc, yloc, diameter, maxHist = 10):
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
    def __init__(self, pupilModelPath=None, faceModelPath='data/haarcascades/haarcascade_frontalface_default.xml', eyeModelPath='data/haarcascades/haarcascade_eye.xml'):
        #TODO try/catch here to prevent runtime errors further on?
        self.faceModel = cv2.CascadeClassifier(faceModelPath)
        self.eyeModel = cv2.CascadeClassifier(eyeModelPath)
        if pupilModelPath == None:
            detector_params = cv2.SimpleBlobDetector_Params()
            detector_params.filterByArea = True
            detector_params.maxArea = 1500
            self.pupilModel = cv2.SimpleBlobDetector_create(detector_params)
        self.pupilModel = pupilModelPath #TODO change this

    def find_face(self, cv2Image):
        faces = self.faceModel.detectMultiScale(cv2Image, 1.3, 5)
        if len(faces) == 0:
            return cv2Image
        x, y, w, h = faces[np.argmax(faces, 0)[2]].tolist() #get biggest detected face, measured by width
        # cv2.rectangle(resultImage,  (x-10, y-20),
        #             (x + w+10 , y + h+20),
        #             (0,165,255),2)
        return cv2Image[y:y+h, x:x+w]
    
    def find_eyes(self, cv2Image):
        #TODO simple data assumptions? eyes will only be in the top half of the model, for e.g. - simplify computation
        eyes = self.eyeModel.detectMultiScale(cv2Image, 1.3, 5)
        resultImage = cv2Image.copy()
        if len(eyes) == 0:
            return cv2Image
        x, y, w, h = eyes[np.argmax(eyes, 0)].tolist()[0] #returns rightmost eye detected (ideally there are only two)
        # cv2.rectangle(resultImage,  (x-10, y-20),
        #             (x + w+10 , y + h+20),
        #             (0,165,255),2)
        return cv2Image[y:y+h, x:x+w]
    
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def preprocess(self, cv2Image, method):
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
        img = exposure.rescale_intensity(img, in_range='image', out_range=(0,255)).astype(np.uint8)
        return img

    def oldPreprocess(self, cv2Image): #TODO DELETE THIS WHEN I'M HAPPY WITH PREPROCESS
        #no doctoring
        pre = [cv2Image.copy()]
        preLabels = ["none"]
        #gaussian blur
        img_blur = cv2.GaussianBlur(cv2Image, (3, 3), sigmaX=0, sigmaY=0)
        pre.append(img_blur.copy())
        preLabels.append("blur")
        #binarizing / thresholding
        _, lowThresh = cv2.threshold(cv2Image, 45, 255, cv2.THRESH_BINARY)
        _, midThresh = cv2.threshold(img_blur, 127, 255, 0)
        _, highThresh = cv2.threshold(img_blur, 200, 255, cv2.THRESH_BINARY)
        pre += [lowThresh.copy(), midThresh.copy(), highThresh.copy()]
        preLabels += ['lowThresh', 'midThresh', 'highThresh']
        # erosion
        binr = cv2.threshold(cv2Image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] 
        kernel = np.ones((5, 5), np.uint8) 
        invert = cv2.bitwise_not(binr) 
        erosion = cv2.erode(invert, kernel, 
                            iterations=1) 
        pre.append(erosion)
        preLabels.append("erosion")
        # dilation
        kernel = np.ones((3, 3), np.uint8) 
        invert = cv2.bitwise_not(binr) 
        dilation = cv2.dilate(invert, kernel, iterations=1) 
        pre.append(dilation)
        preLabels.append("dilation")
        # opening
        kernel = np.ones((3, 3), np.uint8) 
        opening = cv2.morphologyEx(binr, cv2.MORPH_OPEN, kernel, iterations=1) 
        pre.append(opening)
        preLabels.append("opening")
        # closing
        kernel = np.ones((3, 3), np.uint8) 
        closing = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, kernel, iterations=1) 
        pre.append(closing)
        preLabels.append("closing")
        # morphological gradient
        kernel = np.ones((3, 3), np.uint8) 
        invert = cv2.bitwise_not(binr) 
        morph_gradient = cv2.morphologyEx(invert, cv2.MORPH_GRADIENT,  kernel) 
        pre.append(morph_gradient)
        preLabels.append("morph_gradient")
        # black hat
        kernel = np.ones((5, 5), np.uint8) 
        invert = cv2.bitwise_not(binr) 
        black_hat = cv2.morphologyEx(invert, cv2.MORPH_BLACKHAT, kernel) 
        pre.append(black_hat)
        preLabels.append("black_hat")
        # tophat
        kernel = np.ones((13, 13), np.uint8) 
        top_hat = cv2.morphologyEx(invert, cv2.MORPH_TOPHAT, kernel) 
        pre.append(top_hat)
        preLabels.append("top_hat")
        # Edge Detection
        sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
        sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
        sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
        canny = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) 
        pre += [sobelx.copy(), sobely.copy(), sobelxy.copy(), canny.copy()]
        preLabels += ['sobelx', 'sobely', 'sobelxy', 'canny']
        return pre, preLabels

    def get_preprocessed_list(self, cv2Image):
        methods = ["none", "blur", "threshLow", "threshMid", "threshHigh", "erosion", "dilation", "opening", "closing", "morphGradient", "blackHat", "topHat", "sobelx", "xobely", "sobelxy", "canny", "laplacian"]
        methodLabels = []
        methodOutputs = []
        for method1 in methods:
            img = self.preprocess(cv2Image, method1)
            methodOutputs.append(img)
            methodLabels.append(method1)
            # for method2 in methods:
            #     out = self.preprocess(img.copy(), method2)
            #     methodOutputs.append(out)
            #     methodLabels.append(f"{method1}>{method2}")
        return methodOutputs, methodLabels
        
    def find_pupil(self, cv2Image):
        #TODO automatically determine threshold and track threshold between dudes. If lighting changes over the course of the show this will not work. Might be able to assume size and shape and dynamically determine threshold from that??
        pre, preLabels = self.get_preprocessed_list(cv2Image)
        post = []
        postLabels = []

        #setup for final stage
        #template matching
        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        templates = [cv2.imread('data/pupil_template.png', cv2.IMREAD_GRAYSCALE), cv2.imread('data/pupil_template_grey.jpg', cv2.IMREAD_GRAYSCALE)]
        templateLabels = ["png", "jpg"]
        w, h = templates[0].shape[::-1]

        #final step
        for index, img in enumerate(pre):
            #contour detection
            imgCpy = img.copy()
            contours, hierarchy = cv2.findContours(imgCpy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            imgCpy = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(imgCpy, contours, -1, (255,0,0), 2)
            # cv2.imshow("contours", cv2.resize(imgCpy, (192, 192)))
            # cv2.waitKey(0)
            post.append(imgCpy.copy())
            postLabels.append(f"{preLabels[index]} findContours")

            # Apply Hough transform on the blurred image - search for circles
            imgCpy = img.copy()
            detected_circles = cv2.HoughCircles(imgCpy,  
                            cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
                        param2 = 30, minRadius = 1, maxRadius = 40) 
            # Draw circles that are detected. 
            if detected_circles is not None: 
                # Convert the circle parameters a, b and r to integers. 
                detected_circles = np.uint16(np.around(detected_circles)) 
                for pt in detected_circles[0, :]: 
                    imgCpy = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    a, b, r = pt[0], pt[1], pt[2] 
                    # Draw the circumference of the circle. 
                    cv2.circle(imgCpy, (a, b), r, (255, 0, 0), 2) 
                    # Draw a small circle (of radius 1) to show the center. 
                    cv2.circle(imgCpy, (a, b), 1, (0, 255, 0), 3) 
                    # cv2.imshow("imgCpy", cv2.resize(imgCpy, (192, 192)))
                    # cv2.waitKey(0)
                    post.append(imgCpy.copy())
                    postLabels.append(f"{preLabels[index]} detectCircles")

            #template matching
            for meth in methods:
                for i, template in enumerate(templates):
                    imgCpy = img.copy()
                    method =eval(meth)
                    # Apply template Matching
                    res = cv2.matchTemplate(imgCpy,template,method)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                        top_left = min_loc
                    else:
                        top_left = max_loc
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    imgCpy = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    cv2.rectangle(imgCpy,top_left, bottom_right, (255, 0, 0), 5)
                    # cv2.imshow("imgCpy", cv2.resize(imgCpy, (192, 192)))
                    # cv2.waitKey(0)
                    post.append(imgCpy.copy())
                    postLabels.append(f"{preLabels[index]} templateMatching {meth} {templateLabels[i]}")

        #DISPLAY
        pres = cv2.vconcat(pre)
        cv2.imshow((', ').join(preLabels), pres)
        cv2.waitKey(0)
        
        labels = []
        i, prefix, l = 0, 'bwah', []
        while i < len(post):
            if prefix != postLabels[i].split(" ")[0] or i == len(post) - 1:
                if len(l) > 0:
                    posts = cv2.hconcat(l)
                    cv2.imshow(prefix, posts)
                    cv2.waitKey(0)
                    print(" | ".join(postLabels))
                prefix = postLabels[i].split(' ')[0]
                l = []
                labels = []
            else:
                l.append(post[i])
                labels.append(postLabels[i])
            i += 1

        # n = 14
        # chunks = [post[i:i + n] for i in range(0, len(post), n)]
        # chunkLabels = [postLabels[i:i + n] for i in range(0, len(postLabels), n)]
        # # chunksConcat = []
        # for i, chunk in enumerate(chunks):
        #     posts = cv2.hconcat(chunk)
        #     print((', ').join(chunkLabels[i]))
        #     cv2.imshow((', ').join(chunkLabels[i]), posts)
        #     cv2.waitKey(0)
        #     # chunksConcat.append(cv2.hconcat(chunk))

        cv2.imshow("hi", img)
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()
        # pysource eye motion tracking opencv with python
        # medium also does it
        #TODO check how stable the eye tracker is OR find eye edges (ideally both)

def main():
    # testing.
    # https://www.guidodiepen.nl/2017/02/detecting-and-tracking-a-face-with-python-and-opencv/
    baseImage = cv2.imread('data/images/genericFace.png')
    resultImage = baseImage.copy()
    gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
    tracker = Tracker()
    face = tracker.find_face(gray)
    eye = tracker.find_eyes(face)
    pupil = tracker.find_pupil(eye)
    

if __name__ == "__main__":
    main()

