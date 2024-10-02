from FaceMeshModule import FaceMeshDetector
import cv2
import time



if __name__ == '__main__':
    print("Starting...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        cap.release()
        quit()
    
    print("Creating FaceMeshDetector...")
    detector = FaceMeshDetector()
    print("FaceMeshDetector is created!")

    pTime = 0
    ct = 0

    while True:
        if ct == 0:
            print("Frame cap start...")
        ct += 1

        ret, img = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        if ct == 1:
            print("Got first frame!")

        img, faces, landmarks = detector.FindFaceMesh(img)
        
        if ct == 1 and len(faces) != 0:
            print("Faces: ", faces)
            print("Landmarks: ", landmarks)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        
        if ct == 1:
            print("First frame done processing!")
