#coding=utf8


import os
from pdf2image import convert_from_path, convert_from_bytes
import cv2
import numpy as np

from skimage import measure,morphology
from skimage.segmentation import clear_border
from skimage import transform
from skimage import feature

import scipy.cluster.vq as vq

import scipy.ndimage as sciimage


from skimage import color
from skimage.morphology import closing, square
from skimage.filters import threshold_otsu


from skimage.morphology import closing, square
from skimage.color import label2rgb
import matplotlib.patches as mpatches



import matplotlib.pyplot as plt
import pdb

# bin


def preprocess( image ):

    dat = np.mean( image, axis=2).astype(np.uint8)
    print(dat.shape)

    #dat = cv2.blur(dat, (3, 3))
    dat = 255 * (dat > 250).astype(np.uint8)
    # thr, dat = cv2.threshold( dat, 0, 255, cv2.THRESH_OTSU )

    img_h, img_w = dat.shape
    normal_w = 4096
    normal_h = img_h * normal_w / img_w
    dat = cv2.resize(dat, (normal_w, normal_h), interpolation=cv2.INTER_AREA)
    dat = 255 * (dat > 250).astype(np.uint8)
    return dat


def pdf2image( filename ):
    '''
    '''
    imgs = convert_from_path( filename,dpi=1000)
    np_imgs = []

    for a in imgs:
        dat = np.array( a, np.uint8 )
        print(dat.shape)

        dat = np.mean( dat,axis=2 ).astype(np.uint8)
        print(dat.shape)

        dat = cv2.blur( dat,(3,3))
        dat = 255*(dat>250).astype(np.uint8)
        #thr, dat = cv2.threshold( dat, 0, 255, cv2.THRESH_OTSU )

        img_h,img_w = dat.shape
        normal_w = 4096
        normal_h = img_h * normal_w/img_w
        dat = cv2.resize( dat,(normal_w,normal_h ),interpolation=cv2.INTER_AREA )
        dat = 255 * (dat > 250 ).astype(np.uint8)
        #thr, dat = cv2.threshold(dat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        #dat = cv2.Canny(dat, 50, 150)


        np_imgs.append( dat )


    return np_imgs


def seg_img( img ):

    def filter( dat,thr ):
        return dat*(dat<thr)

    img = img<200
    img_h,img_w = img.shape[0:2]

    proj_h = np.sum(img, axis=0)
    proj_w = np.sum(img, axis=1)

    proj_h = filter( proj_h, img_w*0.5 )
    proj_w = filter( proj_w, img_h*0.5 )




    plt.subplot(2,1,1)
    plt.plot( proj_h )

    plt.subplot(2, 1, 2)
    plt.plot( proj_w )

    return proj_h,proj_w

def crop_major_compent( image ):
    '''
    :param img:
    :return:
    '''

    bw = image>0
    #cleared = bw
    cleared = clear_border(bw)

    # label image regions
    label_image = measure.label(cleared)

    image_label_overlay = label2rgb(label_image, image=image)

    #fig, ax = plt.subplots(figsize=(10, 6))
    #ax.imshow(image_label_overlay)

    img_h,img_w = image.shape[0:2]

    size_thr = 0.6
    area_thr = 0.5
    max_area_ratio = 0.0
    max_area_label = 0
    max_rect = None

    for region in measure.regionprops(label_image):
        # take regions with large enough areas
        #
        area_ratio = float(region.area) / (img_h * img_w)

        if area_ratio > area_thr and area_ratio>max_area_ratio:

            ymin,xmin,ymax,xmax = region.bbox

            rect_w = xmax - xmin
            rect_h = ymax - ymin

            print( xmin,ymin,xmax,ymax, float(rect_w)/img_w,float(rect_h)/img_h )


            if (rect_w > size_thr * img_w ) and ( rect_h > size_thr * img_h ):

                max_area_ratio = area_ratio
                max_area_label = region.label
                max_rect = region.bbox

    #
    if max_area_ratio<area_thr:
        return None,None


    mask = np.equal( label_image, max_area_label )

    mask = sciimage.morphology.binary_fill_holes( mask )

    #
    new_img = ( mask * image + 255*(mask==0) )

    crop_img = new_img[ max_rect[0]:max_rect[2],max_rect[1]:max_rect[3] ].astype( np.uint8 )

    #pdb.set_trace()

    '''
    plt.subplot(131)
    plt.imshow( mask )

    plt.subplot(132)
    plt.imshow(crop_img,cmap='gray')

    plt.subplot(133)
    plt.imshow(image)

    plt.tight_layout()
    plt.show()
    '''

    return crop_img,max_rect


def detect_lines( image ):
    '''
    :param image:
    :return:
    '''

    #edges = feature.canny(image, sigma=2, low_threshold=1, high_threshold=25)
    edges = image< 5
    edges = morphology.skeletonize(edges)

    img_h,img_w = image.shape[0:2]
    thr = min( img_w,img_h ) / 50

    lines = transform.probabilistic_hough_line(edges, threshold=10, line_length=thr, line_gap=10)

    # 创建显示窗口.
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 6))
    plt.tight_layout()

    # 显示原图像
    ax0.imshow(image, plt.cm.gray)
    ax0.set_title('Input image')
    ax0.set_axis_off()

    # 显示canny边缘
    ax1.imshow(edges, plt.cm.gray)
    ax1.set_title('Canny edges')
    ax1.set_axis_off()

    # 用plot绘制出所有的直线
    ax2.imshow(edges * 0)
    for line in lines:
        p0, p1 = line
        ax2.plot((p0[0], p1[0]), (p0[1], p1[1]))
    row2, col2 = image.shape
    ax2.axis((0, col2, row2, 0))
    ax2.set_title('Probabilistic Hough')
    ax2.set_axis_off()
    plt.show()

    '''
    hspace, angles, dists =  transform.hough_line( image<128 )
    hspace, angles, dists = transform.hough_peaks(hspace, angles, dists,min_angle=15 )


    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - col1 * np.cos(angle)) / np.sin(angle)
    '''

def det_chars( image ):

    bw = image < 10
    label_image = measure.label( bw )

    char_size = 100

    region_w_arr = []
    region_h_arr = []
    region_area_arr = []

    regions = measure.regionprops(label_image)

    char_idx = []

    mask = np.zeros_like( image,dtype=np.int32 )

    for region in regions:
        ymin, xmin, ymax, xmax = region.bbox
        area = region.area

        rect_w = xmax - xmin
        rect_h = ymax - ymin

        region_area_arr.append( area )
        region_h_arr.append( rect_h )
        region_w_arr.append( rect_w )

        if rect_w<char_size and rect_h < char_size and rect_w > 5 and rect_h>5:
            char_idx.append( region.label )

            mask[ ymin:ymax,xmin:xmax ] = label_image[ymin:ymax,xmin:xmax] #region.label




    #hist
    region_w_arr = np.array( region_w_arr )
    region_h_arr = np.array( region_h_arr )
    region_area_arr = np.array( region_area_arr )

    def kmean_proc( a,k ):
        a = np.expand_dims( a,axis=1 ).astype(np.float32 )
        #a = vq.whiten( a )
        return vq.kmeans( a, k )

    centroid_w,dist_w = kmean_proc( region_w_arr, 5 )
    centroid_h, dist_h = kmean_proc( region_h_arr, 5)
    centroid_area, dist_area = kmean_proc( region_area_arr, 5)

    hist_w, hist_w_edge = np.histogram( region_w_arr,bins=5 )
    hist_h, hist_h_edge = np.histogram(region_h_arr, bins=5)
    hist_area, hist_area_edge = np.histogram(region_area_arr, bins=5)



    plt.subplot(121)
    plt.imshow( mask )
    plt.subplot(122)
    plt.imshow( label2rgb(label_image ), )

    fig = plt.figure(2)
    plt.imshow( (image<10) * (mask==0),cmap ='gray' )

    plt.show()







def proc( filename,save_path=None,crop=True ):
    '''
    :return:
    '''

    #imgs = pdf2image( filename )
    #assert( len(imgs)==1 )
    #image = imgs[0]
    image = cv2.imread( filename )
    image = preprocess( image )

    #pdb.set_trace()

    if crop:
        crop_img,region_rect = crop_major_compent( image )
    else:
        crop_img = image

    if crop_img is None:
        print( 'error')
        print( filename )
        return

    #detect_lines( crop_img )

    det_chars( crop_img )


    if save_path is not None:
        save_filename = os.path.join( save_path, os.path.basename( filename ) )
        cv2.imwrite( save_filename,crop_img )

    if 0:
        plt.subplot(121)
        plt.imshow( image,cmap='gray' )
        plt.subplot(122)
        plt.imshow( crop_img, cmap='gray')
        plt.show()



def main():

    list_fn = './test_list2.txt'
    save_path = None #'./debug'

    with open( list_fn ) as fp:
        fn_list = [a.strip() for a in fp.readlines() ]

    cnt = 0
    for a_file in fn_list:

        cnt += 1
        print('proc=%s,%d/%d'%( a_file,cnt,len(fn_list) ) )

        proc( a_file,save_path,crop=False )



if __name__=='__main__':

    main()
