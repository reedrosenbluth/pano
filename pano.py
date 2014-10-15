import sys
import cv2, numpy as np

def extract_features(image, surfThreshold=1000):
    surf = cv2.SURF(surfThreshold)
    surfDescriptorExtractor = cv2.DescriptorExtractor_create("SURF")
    keypoints = surf.detect(image)
    (keypoints, descriptors) = surfDescriptorExtractor.compute(image,keypoints)
    return (keypoints, descriptors)

def find_correspondences(keypoints1, descriptors1, keypoints2, descriptors2):
    match = match_flann(descriptors1, descriptors2)
    p1 = []
    p2 = []
    for m in match:
        p1.append(keypoints1[m[0]].pt)
        p2.append(keypoints2[m[1]].pt)
    points1 = np.array(p1, np.float32)
    points2 = np.array(p2, np.float32)
    return (points1, points2)

def calculate_size(size_image1, size_image2, homography):
    offset = abs((homography*(size_image2[0]-1,size_image2[1]-1,1))[0:2,2]) 
    size = (size_image1[1] + int(offset[0]), size_image1[0] + int(offset[1]))
    if (homography*(0,0,1))[0][1] > 0:
        offset[0] = 0
    if (homography*(0,0,1))[1][2] > 0:
        offset[1] = 0

    return (size, offset)

def merge_images(image1, image2, homography, size, offset, keypoints):
    panorama = cv2.warpPerspective(image2,homography,size)
    (h1, w1) = image1.shape[:2]
    for h in range(h1):
        for w in range(w1):
            try:
                panorama[h][w] = image1[h][w]
            except:
                continue
    return panorama

def match_flann(desc1, desc2, r_threshold = 0.06):
    FLANN_INDEX_KDTREE = 1
    flann = cv2.flann_Index(desc2, {'algorithm': FLANN_INDEX_KDTREE, 'trees': 4})

    (idx2, dist) = flann.knnSearch(desc1, 2, params={}) # bug: need empty {}

    mask = dist[:,0] / dist[:,1] < r_threshold

    idx1  = np.arange(len(desc1))
    pairs = np.int32(zip(idx1, idx2[:,0]))
    return pairs[mask]

def draw_correspondences(image1, image2, points1, points2):
    (h1, w1) = image1.shape[:2]
    (h2, w2) = image2.shape[:2]
    image = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    image[:h1, :w1] = image1
    image[:h2, w1:w1+w2] = image2

    for (x1, y1), (x2, y2) in zip(np.int32(points1), np.int32(points2)):
        cv2.line(image, (x1, y1), (x2+w1, y2), (0, 255, 255), lineType=cv2.CV_AA)

    return image

if __name__=="__main__":

    image1 = cv2.imread(sys.argv[1])
    image2 = cv2.imread(sys.argv[2])

    (keypoints1, descriptors1) = extract_features(image1)
    (keypoints2, descriptors2) = extract_features(image2)
    print len(keypoints1), "features detected in image1"
    print len(keypoints2), "features detected in image2"

    (points1, points2) = find_correspondences(keypoints1, descriptors1, keypoints2, descriptors2)
    print len(points1), "features matched"

    correspondences = draw_correspondences(image1, image2, points1, points2)
    cv2.imwrite("correspondences.jpg", correspondences)

    (homography, _) = cv2.findHomography(points2, points1)
    print homography

    (size, offset) = calculate_size(image1.shape, image2.shape, homography)
    print "output size: %ix%i" % size

    panorama = merge_images(image1, image2, homography, size, offset, (points1, points2))
    cv2.imwrite("panorama.jpg", panorama)



