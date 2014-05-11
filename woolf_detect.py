import sys, getopt
import os, os.path
import cPickle as pickle
import cv2

surf = cv2.SURF(1000, upright=0)            # TODO: make hessian factor config'able
bf_matcher = cv2.BFMatcher()

def do_training(training_dir, output_dir):
    if output_dir == "":
        print "Training mode requires output directory (specify using -o or --output)"
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    files_to_proc = []
    for root, dirs, files in os.walk(training_dir):
        for file in files:
            full_path = os.path.join(root, file)
            files_to_proc.extend([full_path])

    kps = []
    descs = []
    for full_path in files_to_proc:
        print "Processing " + full_path
        img = cv2.imread(full_path, 0)
        kp, des = surf.detectAndCompute(img, None)
        
        kps.extend([kp])
        descs.extend([des])
        
        ## NOTE: we don't really need this for the classification but it's just something cool(ish) to look at
        img_kp = cv2.drawKeypoints(img, kp, None, (0,0,255), 4)
        _, file = os.path.split(full_path)
        if not cv2.imwrite(os.path.join(output_dir, file), img_kp):
            print "Unable to save " + os.path.join(output_dir, file) + "!"
            
	
    save_keypoints_descriptors(kps, descs, output_dir)
    print "Training done!"

def save_keypoints_descriptors(all_kps, all_descs, output_dir):
    if len(all_kps) != len(all_descs):
        print "Length of keypoints and descriptors don't match!"
        return
        
    kp_descs_pickle = [] 
    for i in range(len(all_kps)):
        kps = all_kps[i]
        descs = all_descs[i]
        kp_desc = []
        for j in range(len(kps)):
            # Combine keypoints data and descriptor into one object
            temp = (kps[j].pt, kps[j].size, kps[j].angle, kps[j].response, kps[j].octave, kps[j].class_id, descs[j])
            kp_desc.append(temp)
        print "Saving " + str(len(kps)) + " keypoints-descriptor pairs for image " + str((i + 1))
        kp_descs_pickle.append(kp_desc)
        
    keypoints_db = os.path.join(output_dir, "woolf.kpd")
    try:
        pickle.dump(kp_descs_pickle, open(keypoints_db, "wb"))
        print "Keypoints database saved to " + keypoints_db
    except IOError:
        print "Could not save to keypoints database file " + keypoints_db
    
    
def load_keypoints_descriptors(keypoints_db):
    try:
        kp_descs_pickle = pickle.load(open(os.path.join(keypoints_db), "rb" ))
    except IOError:
        print "Could not open keypoints database file " + keypoints_db
        return [], []

    all_kps = []
    all_descs = []
    i = 0
    for kp_desc in kp_descs_pickle:
        kps = []
        descs = []
        for temp in kp_desc:
            kp = cv2.KeyPoint(x=temp[0][0],y=temp[0][1],_size=temp[1], _angle=temp[2], _response=temp[3], _octave=temp[4], _class_id=temp[5])
            desc = temp[6]
            kps.append(kp)
            descs.append(desc)
        all_kps.append(kps)
        all_descs.append(descs)
        print "Read " + str(len(kps)) + " keypoints-descriptor pairs for image " + str((i + 1))
        i+=1
    
    return all_kps, all_descs    
	
def do_classify(test_dir, keypoints_db, training_dir, output_dir):
    if output_dir == "":
        print "Classify mode requires output directory (specify using -o or --output)"
        return
        
    if keypoints_db == "" and training_dir == "":
        print "Either a keypoints file or a training data directory should be specified (-t or -k)"
        return
        
    if keypoints_db != "":
        all_kps, all_descs = load_keypoints_descriptors(keypoints_db)
    else:
        # TODO do training and then classification
        return
	
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print "Classifying images in " + test_dir
    ## TODO real classification!!!
	

def show_help():
    print "options: "
    print "-t, --training <trainingdatadir>       Enter training mode and use given directory for training data"
    print "-o, --output <outputdir>               Location of output descriptors (in training mode) or classified images (in classify mode)"
    print "-c, --classify <imgdir>                Enter classify mode and classify the images in the given directory"
    print "-k, --keypoints <keypointsfile>        Use given file as source of keypoints"

def main(argv):
    if len(argv) == 0:
        show_help()
        sys.exit()

    # Parse parameters.
    training_dir = ""
    output_dir = ""
    test_dir = ""
    keypoints_db = ""
    training_mode = False
    classify_mode = False
    try:
        opts, args = getopt.getopt(argv,"ht:o:c:k:",["help", "training=","output=", "classify=", "keypoints="])
    except getopt.GetoptError:
        show_help()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            show_help()
            sys.exit()
        elif opt in ("-t", "--training"):
            training_dir = arg
            training_mode = True
        elif opt in ("-o", "--output"):
            output_dir = arg
        elif opt in ("-c", "--classify"):
            test_dir = arg
            classify_mode = True
        elif opt in ("-k", "--keypoints"):
            keypoints_db = arg
    
    if classify_mode:
        do_classify(test_dir, keypoints_db, training_dir, output_dir)
    elif training_mode:
        do_training(training_dir, output_dir)
            
main(sys.argv[1:])
