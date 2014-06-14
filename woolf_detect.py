import sys, getopt
import os, os.path
import shutil
import cPickle as pickle
import numpy as np
import math
import cv2
import pprint
import kmeans

DEFAULT_HESSIAN = 1000
DEFAULT_GMRATIO = 0.8

pp = pprint.PrettyPrinter(indent=4)

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
    # Delete previous keypoints db if it exists
    if os.path.exists(keypoints_db):
        os.remove(keypoints_db)                 
    
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
        return 0, [], []

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
	
def find_best_cluster(desc, centroids):
    best_cluster = 0
    least_distance = None
    for cluster in range(len(centroids)):
        c = centroids[cluster]
        distance = np.linalg.norm(desc-c)
        if least_distance is None or distance < least_distance:
            least_distance = distance
            best_cluster = cluster
        
    return best_cluster
    
def get_hist_difference(query_hist, training_hist):
    ## ?? What's the best way to get the diff between (feature) histograms??
    ## If we treat each histogram as a vector (with dim=# of bins), let's take the euclidean distance 
    total = 0
    for cluster in query_hist:
        sq_diff = (len(query_hist[cluster]) - len(training_hist[cluster]))**2
        #print "Cluster " + str(cluster) + ": " + str(diff) 
        total += sq_diff
    return math.sqrt(total)
    
def do_bof_classification_nn(qimg_pathname, qdescs, centroids, training_hist):
    query_hist = {}
    for qdesc in qdescs:
        best_cluster = find_best_cluster(qdesc, centroids)
        try:
            query_hist[best_cluster].append(qdesc)
        except KeyError:
            query_hist[best_cluster] = [qdesc]
    
    hist_diff = get_hist_difference(query_hist, training_hist)
    print "TOTAL DIFF: " + str(hist_diff)
    yes_classify = (hist_diff < 195.0)   ## some threshold value
    if yes_classify:
        print "+++ Image " + qimg_pathname + " is a POSITIVE!"
                
    return yes_classify 
    
def do_bruteforce_classification(qimg_pathname, qdescs, all_descs, bf, min_num_hits):
    i = 1
    num_hits = 0
    for trdescs in all_descs:
        print "Trying with descriptor" + str(i) + " (" + str(len(trdescs)) + " descriptors)" 
        ## Find the best k matches between training descriptor and query descriptor
        ## NOTE: need to do np.asarray() in order for the function to work -- maybe a python version issue
        ## Need to understand this better -- what exactly is being matched? What is the structure of the descriptors??
        ## Each "descriptor" is actually an array of 128 float32 values -- it's SIFT/SURF's numerical representation of a feature
        ## The keypoint structure contains metadata about the feature -- it's x,y location, octave, angle, etc.
        ## If we treat each descriptor as a 128-d vector, knnMatch will then attempt to find the k nearest descriptor vectors
        ## Note that qdesc is a set of descriptors, with each descriptor represented by 128 values 
        ## For each descriptor q in qdesc, bf.knnMatch will attempt to find the k nearest neighbors of q in trdesc 
        matches = bf.knnMatch(np.asarray(qdescs, np.float32),
                              np.asarray(trdescs, np.float32),k=2)
        ## The size of matches will then be equal to the size of the query descriptor set, BUT each element in the list has k (or less) match objects
        ##print "len(matches): " + str(len(matches))
        ##pp = pprint.PrettyPrinter(indent=4)
        ##pp.pprint(matches)
        
        ## What is the structure of this matches list?
        ## Each element in the matches list contains k (possibly less) "match" objects.
        ## Each "match" contains queryIdx, trainIdx, imgIdx, and distance (between the descriptor qdesc[queryIdx] and trdesc[trainIdx])
        
        # Apply ratio test
        ## The goal of this is to determine whether the distances of a pair of match objects is less than some threshold (in DLowe SIFT paper)
        ## Since k=2, m and n will be two match objects for the same query image descriptor (i.e. m.queryIdx == n.queryIdx) but with two different
        ## training image descriptors (i.e. m.trainIdx != n.trainIdx)
        good_matches = 0
        for m,n in matches:
            if m.distance < DEFAULT_GMRATIO*n.distance:
                good_matches+=1
                
        num_min_matches = math.floor(max(len(qdescs), len(trdescs)) * 0.065)   # 6.5% of the descriptors in either query or train image
        print "Found " + str(good_matches) + " good matches (" + str(num_min_matches) + " required)" 
    
        # Only consider this image to be of class 1 (HAS LOGO) if there are enough good matches
        ## Possible refinement: consider query image as class 1 only if there are 
        ## at least j matches (1 <= j <= #trainingimgs)? No theoretical basis, but it's something :D
        ## Perhaps num_min_matches should be a proportion of the number of features extracted from query and current training desc?
        ## Also NUM_HITS should be a function of the number of training images read from the keypoints db??
        ## Because query images with large number of descs (>1000) tend to be classified as YES LOGO
        if good_matches >= num_min_matches:
            num_hits+=1
            if num_hits == min_num_hits:
                print "+++ Image produced " + str(num_hits) + " hits; " + qimg_pathname + " is a POSITIVE!"
                return True
        i+=1
            
    return False

def do_classify(test_dir, keypoints_db, output_dir, results_prefix, classify_mode_alg=1):
    if output_dir == "":
        output_dir = test_dir + "_out"
        
    if keypoints_db == "":
        print "A keypoints database should be specified (specify using -k or --keypoints)"
        return
 
    hessian, all_kps, all_descs = load_keypoints_descriptors(keypoints_db)
    if len(all_descs) == 0:
        print "No keypoints/descriptors loaded"
        return
    
    ## TODO this should be done in training stage and saved to file      
    ## try to find k=50 clusters for the set of training descriptors that we have
    ## flatten list of descs
    if classify_mode_alg == 2:
        all_descs2 = [item for sublist in all_descs for item in sublist]
        centroids, training_hist = kmeans.cluster(all_descs2, 50)
        ## TODO during training, do CH index comparison to find the "best" set of clustering
        print "Done finding " + str(len(training_hist)) + " clusters in training set"
        #for cluster in clusters:
        #    print "Cluster " + str(cluster) + ": " + str(len(clusters[cluster])) + " points" 
        #print "Centroids: " 
        #pp.pprint(mu)
        
    if classify_mode_alg == 1:
        min_num_hits = len(all_descs)/3
        if min_num_hits == 0:
            min_num_hits = 1
        print "Images must have at least " + str(min_num_hits) + " hits for a positive classification"
        
    if os.path.exists(output_dir):
        ## Delete its contents!
        print "Removing " + output_dir + " prior to classification..."
        shutil.rmtree(output_dir)
        
    output_dir_yes = os.path.join(output_dir, "yes")
    output_dir_no = os.path.join(output_dir, "no")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir_yes):    
        os.makedirs(output_dir_yes)
    if not os.path.exists(output_dir_no):    
        os.makedirs(output_dir_no)

    files_to_proc = get_image_files(test_dir)
    if len(files_to_proc) == 0:
        print "Found no image files in " + test_dir
        return

    surf = cv2.SURF(hessian, upright=0)            
    bf = cv2.BFMatcher(cv2.NORM_L2)
            
    for qimg_pathname in files_to_proc:
        print "\n-------------------------------------\nClassifying " + qimg_pathname
        _, qimg_filename = os.path.split(qimg_pathname)    
        
        qimg = cv2.imread(qimg_pathname, 0)
                                
        ## Get the descriptors of the query image
        _, qdescs = surf.detectAndCompute(qimg, None)
        print "Found " + str(len(qdescs)) + " descriptors in query image"
    
        yes_classify = False
        if classify_mode_alg == 1:
            yes_classify = do_bruteforce_classification(qimg_pathname, qdescs, all_descs, bf, min_num_hits)
        else:
            yes_classify = do_bof_classification_nn(qimg_pathname, qdescs, centroids, training_hist)
        
        dest_path = os.path.join(output_dir_no, qimg_filename)
        if yes_classify:
            dest_path = os.path.join(output_dir_yes, qimg_filename)
        
        shutil.copyfile(qimg_pathname, dest_path)
	
    if results_prefix != "":
        show_classification_results(output_dir_yes, output_dir_no, results_prefix)

def show_classification_results(output_dir_yes, output_dir_no, prefix):
    yes_yes, yes_no = count_filename_matches(output_dir_yes, prefix)
    no_yes, no_no = count_filename_matches(output_dir_no, prefix)
    
    print "\n-=-=-=-=-=- CLASSIFICATION RESULTS -=-=-=-=-=-"
    print "Results were computed by checking if the filename starts with \"" + prefix + "\""
    print "Column headers represent the truth, row headers represent the predictions done by the classifier"
    print "%8s" % "YES" + "%5s" % "NO"
    print "YES %4s" % str(yes_yes) + "%5s" % str(yes_no)
    print "NO %5s" % str(no_yes) + "%5s" % str(no_no)
    
    ## Compute accuracy
    total = yes_yes + yes_no + no_yes + no_no
    correct = yes_yes + no_no
    pct_correct = (correct/float(total)) * 100.0
    pct_error = 100.0 - pct_correct
    print "\nCorrectly classified: " + str(correct) + "/" + str(total) + " (%.2f" % pct_correct + "%% accuracy, %.2f" % pct_error + "% error rate)"
            
def count_filename_matches(dir, prefix):
    yes_match = 0
    no_match = 0
    ## Get list of filenames in the given directory
    pathnames = get_image_files(dir)
    ## For each filename, check if it starts with the given prefix,
    ## and increment counters based on the result
    for pname in pathnames:
        _, fname = os.path.split(pname)
        if fname.lower().startswith(prefix):
            yes_match += 1
        else:
            no_match += 1
    
    return yes_match, no_match            
            
def show_help():
    print "options: "
    print "-t, --training <training-data-dir>       Enter training mode and use given directory for training data"
    print "-o, --output <output-dir>                Location of output descriptors (in training mode) or classified images (in classify mode) -- default is no output in training mode, <img-dir>_out in classify mode"
    print "-1, --classify1 <img-dir>                Enter classify 1 mode (using brute force matcher) and classify the images in the given directory"
    print "-2, --classify2 <img-dir>                Enter classify 2 mode (using image histogram matcher) and classify the images in the given directory"
    print "-k, --keypoints <keypoints-file>         Use given file as source of keypoints"
    print "-h, --hessian <hessian-value>            Hessian threshold value (only used when training -- default " + str(DEFAULT_HESSIAN) + ")"
    print "-r, --results <prefix-value>             Check results after classification by inspecting filenames (filename that starts with the given prefix means it contains the logo)"
    
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
    classify_mode_alg = 0
    hessian = DEFAULT_HESSIAN
    results_prefix = ""
    try:
        opts, args = getopt.getopt(argv,"h:t:o:1:2:k:r:",["hessian=", "training=","output=", "classify1=", "classify2=", "keypoints=", "results="])
    except getopt.GetoptError:
        show_help()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-t", "--training"):
            training_dir = arg
            training_mode = True
        elif opt in ("-o", "--output"):
            output_dir = arg
        elif opt in ("-1", "--classify1"):
        
            test_dir = arg
            classify_mode = True
            classify_mode_alg = 1
        elif opt in ("-2", "--classify2"):
            test_dir = arg
            classify_mode = True
            classify_mode_alg = 2
        elif opt in ("-k", "--keypoints"):
            keypoints_db = arg
        elif opt in ("-r", "--results"):
            results_prefix = arg
        elif opt in ("-h", "--hessian"):
            try:
                hessian = arg
            except ValueError:
                print "Illegal value for -h/--hessian: " + arg
                sys.exit(3)
            
    if not classify_mode and not training_mode:
        show_help()
        sys.exit(1)
    
    if classify_mode:
        do_classify(test_dir, keypoints_db, output_dir, results_prefix, classify_mode_alg)
    elif training_mode:
        do_training(training_dir, output_dir, hessian)
            
main(sys.argv[1:])
