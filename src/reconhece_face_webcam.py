import face_recognition
import imutils
import pickle
import cv2

output = None
display = 0

# load the known faces and embeddings
print("[INFO] loading encodings...")
# data = pickle.loads(open(args["encodings"], "rb").read())
data = pickle.loads(open("encod.pickle", "rb").read())

# initialize the pointer to the video file and the video writer
print("[INFO] processing video...")
# stream = cv2.VideoCapture(args["input"])
stream = cv2.VideoCapture(0)
writer = None
# loop over frames from the video file stream
while True:
    # grab the next frame
    (grabbed, frame) = stream.read()
    # print("FRAME: ", frame) 

    # if the frame was not grabbed, then we have reached the
    # end of the stream
    if not grabbed:
        break;

    # convert the input frame from BGR to RGB then resize it to have
    # a width of 750px (to speedup processing)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])
    # print("Antes do boxes")
    boxes = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    # print("Encodings ", encodings) 

    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # rescale the face coordinates
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        # draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)

    # if the video writer is None *AND* we are supposed to write
    # the output video to disk initialize the writer

    if writer is None and output is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output, fourcc, 24,
                                 (frame.shape[1], frame.shape[0]), True)

    # if the writer is not None, write the frame with recognized
    # faces t odisk
    if writer is not None:
        writer.write(frame)

    # check to see if we are supposed to display the output frame to
    # the screen
    if display > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# close the video file pointers
stream.release()

# check to see if the video writer point needs to be released
if writer is not None:
	writer.release()
