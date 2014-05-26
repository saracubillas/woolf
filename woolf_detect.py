import sys, getopt
import os, os.path
import shutil
import cPickle as pickle
import numpy as np
import cv2

DEFAULT_HESSIAN = 1000
DEFAULT_GMRATIO = 0.8
DEFAULT_MIN_MATCHES = 20

def is_image_file(filename):
    _, ext = os.path.splitext(filename)
    if ext.lower() in ['.jpg', '.png', '.jpeg']:
        return True
    return False

def get_image_files(dir):
    file_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            # Only include files whose extension is a known image type
            if is_image_file(file):
                full_path = os.path.join(root, file)
                file_list.extend([full_path])
    return file_list
    
    
def do_training(training_dir, output_dir = "", hessian = DEFAULT_HESSIAN):

    if output_dir != "" and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    files_to_proc = get_image_files(training_dir)
    if len(files_to_proc) == 0:
        print "Found no image files in " + training_dir
        return

    all_kps = []
    all_descs = []
    surf = cv2.SURF(hessian, upright=0)            
    for full_path in files_to_proc:
        print "Processing " + full_path
        img = cv2.imread(full_path, 0)
        kp, des = surf.detectAndCompute(img, None)
        
        all_kps.extend([kp])
        all_descs.extend([des])
        
        if output_dir != "":
            ## NOTE: we don't really need this for the classification but it's just something cool(ish) to look at
            img_kp = cv2.drawKeypoints(img, kp, None, (0,0,255), 4)
            _, file = os.path.split(full_path)
            if not cv2.imwrite(os.path.join(output_dir, file), img_kp):
                print "Unable to save " + os.path.join(output_dir, file) + "!"
            
    save_keypoints_descriptors(hessian, all_kps, all_descs, training_dir)
    print "Training done!"
    return all_kps, all_descs

def save_keypoints_descriptors(hessian, all_kps, all_descs, output_dir):
    if len(all_kps) != len(all_descs):
        print "Length of keypoints and descriptors don't match!"
        return
        
    kp_descs_pickle = [hessian] 
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
    print "Reading from keypoints database file " + keypoints_db
    try:
        kp_descs_pickle = pickle.load(open(keypoints_db, "rb" ))
    except IOError:
        print "Could not open keypoints database file " + keypoints_db
        return [], []

    hessian = kp_descs_pickle[0]
    print "Read hessian value: " + str(hessian)
    kp_descs_pickle = kp_descs_pickle[1:]
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
    
    return hessian, all_kps, all_descs    
	
def do_classify(test_dir, keypoints_db, output_dir,  
                num_min_matches = DEFAULT_MIN_MATCHES, gm_ratio = DEFAULT_GMRATIO):
    if output_dir == "":
        print "Classify mode requires output directory (specify using -o or --output)"
        return
        
    if keypoints_db == "":
        print "A keypoints database should be specified (specify using -k or --keypoints)"
        return
 
    hessian, all_kps, all_descs = load_keypoints_descriptors(keypoints_db)
    
    output_dir_yes = os.path.join(output_dir, "yes_logo")
    output_dir_no = os.path.join(output_dir, "no_logo")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir_yes):    
        os.makedirs(output_dir_yes)
    if not os.path.exists(output_dir_no):    
        os.makedirs(output_dir_no)

    print "Images must have at least " + str(num_min_matches) + " good matches to produce a positive classification" 
    files_to_proc = get_image_files(test_dir)
    if len(files_to_proc) == 0:
        print "Found no image files in " + test_dir
        return

    surf = cv2.SURF(hessian, upright=0)            
    bf = cv2.BFMatcher(cv2.NORM_L2)
            
    for qimg_pathname in files_to_proc:
        print "Classifying " + qimg_pathname
        _, qimg_filename = os.path.split(qimg_pathname)    
        
        qimg = cv2.imread(qimg_pathname, 0)
                                
        ## Get the descriptors of the query image
        _, qdesc = surf.detectAndCompute(qimg, None)
        print("Found " + str(len(qdesc)) + " descriptors in query image")
        
        i = 1
        yes_logo = False
        for trdesc in all_descs:
            print "Trying with descriptor" + str(i) + " (" + str(len(trdesc)) + " descriptors)" 
            ## Find the best k matches between training descriptor and query descriptor
            ## NOTE: need to do np.asarray() in order for the function to work -- maybe a python version issue
            matches = bf.knnMatch(np.asarray(qdesc, np.float32),
                                  np.asarray(trdesc, np.float32),k=2)
                       
            # Apply ratio test
            good_matches = 0
            for m,n in matches:
                if m.distance < gm_ratio*n.distance:
                    good_matches+=1
                    
            print "Found " + str(good_matches) + " good matches " + str(good_matches >= num_min_matches)
            # Only consider this image to be of class 1 if there are enough good matches
            if good_matches >= num_min_matches:
                print "+++ " + qimg_pathname + " has LOGO!"
                yes_logo = True
                shutil.copyfile(qimg_pathname, os.path.join(output_dir_yes, qimg_filename))
                break
            i+=1
            
        if not yes_logo:
            shutil.copyfile(qimg_pathname, os.path.join(output_dir_no, qimg_filename))
	

def show_help():
    print "options: "
    print "-t, --training <training-data-dir>       Enter training mode and use given directory for training data"
    print "-o, --output <output-dir>                Location of output descriptors (in training mode) or classified images (in classify mode)"
    print "-c, --classify <img-dir>                 Enter classify mode and classify the images in the given directory"
    print "-k, --keypoints <keypoints-file>         Use given file as source of keypoints"
    print "-h, --hessian <hessian-value>            Hessian threshold value (only used when training -- default " + str(DEFAULT_HESSIAN) + ")"
    print "-m, --minmatches <num-minimum>           Minimum number of good descriptor matches for a positive classification (only used when classifying -- default " + str(DEFAULT_MIN_MATCHES) + ")"

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
    hessian = DEFAULT_HESSIAN
    minmatches = DEFAULT_MIN_MATCHES
    try:
        opts, args = getopt.getopt(argv,"h:t:o:c:k:m:",["hessian=", "training=","output=", "classify=", "keypoints=", "minmatches="])
    except getopt.GetoptError:
        show_help()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-t", "--training"):
            training_dir = arg
            training_mode = True
        elif opt in ("-o", "--output"):
            output_dir = arg
        elif opt in ("-c", "--classify"):
            test_dir = arg
            classify_mode = True
        elif opt in ("-k", "--keypoints"):
            keypoints_db = arg
        elif opt in ("-h", "--hessian"):
            try:
                hessian = arg
            except ValueError:
                print "Illegal value for -h/--hessian: " + arg
                sys.exit(3)
            
        elif opt in ("-m", "--minmatches"):
            try:
                minmatches = int(arg)
            except ValueError:
                print "Illegal value for -m/--minmatches: " + arg
                sys.exit(3)
            
    if not classify_mode and not training_mode:
        show_help()
        sys.exit(1)
    
    if classify_mode:
        do_classify(test_dir, keypoints_db, output_dir, num_min_matches = minmatches)
    elif training_mode:
        do_training(training_dir, output_dir, hessian)
            
main(sys.argv[1:])
