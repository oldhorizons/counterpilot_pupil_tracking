import cv2
import pypupilext as pp
from matplotlib import pyplot as plt
from constants import ip, http_port, osc_ip, osc_port
from pythonosc import udp_client
from imageProcessing import Tracker
    
class PupilTracker:
    def __init__(self, osc_ip, osc_port):
        self.model = pp.PuReST()
        self.models = [pp.ElSe(), pp.ExCuSe(), pp.PuRe(), pp.PuReST(), pp.Starburst(), pp.Swirski2D()]
        self.ROIs = {}
        self.pupils = {}
        self.visualiser = []  #TODO
        self.oscClient = udp_client.SimpleUDPClient(osc_ip, osc_port)
        print(f"initialised OSC client to {osc_ip}:{osc_port}")
        self.tracker = Tracker()

    def draw_pupil_and_show(self, cv2Image, pupil): 
        x, y = pupil.center
        x = int(x)
        y = int(y)
        d = pupil.majorAxis()
        print(str((pupil.majorAxis(), pupil.minorAxis())))
        if d < 0:
            plt.imshow(cv2Image)
            plt.show()
            print("no pupil detected.")
            return
        colourImg = cv2.cvtColor(cv2Image, cv2.COLOR_GRAY2RGB)
        #draw circumference
        cv2.circle(colourImg, (x, y), d//2, (255, 0, 0), 1)
        #draw center
        cv2.circle(colourImg, (x, y), 1, (0, 255, 0), 1) 
        plt.imshow(colourImg)
        plt.show()
    
    def send_pupil(self, pupil, clientID):
        x, y = pupil.center
        self.oscClient.send_message(f"/cue/eye{clientID}D/name", pupil.majorAxis())
        self.oscClient.send_message(f"/cue/eye{clientID}X/name", x)
        self.oscClient.send_message(f"/cue/eye{clientID}Y/name", y)
        print(f"sent pupil {clientID}")

    def process_pupil(self, cv2Image, clientID):
        img = cv2.cvtColor(cv2Image, cv2.COLOR_RGB2GRAY)
        #if this is the first time seeing this pupil, initialise.
        if clientID not in self.ROIs.keys():
            cv2.namedWindow("Select Area of Interest", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Select Area of Interest", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            x, y, w, h = cv2.selectROI("Select Area of Interest", img)
            self.ROIs[clientID] = [x, y, w, h]
            self.pupils[clientID] = []
            cv2.destroyAllWindows()
            img_crop = img[y:y+h, x:x+w]
            img = img_crop
        for model in self.models:
            pupil = model.run(img)
            print(model)
            self.draw_pupil_and_show(img.copy(), pupil)
        # self.tracker.find_pupil_cv2(img.copy())
        self.pupils[clientID].append(pupil)
        if len(self.pupils[clientID]) > 30:
            self.pupils[clientID] = self.pupils[clientID][:30]
        self.send_pupil(pupil, clientID)
        return self.ROIs[clientID]
