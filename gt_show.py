

import scipy.misc
import numpy as np
import json
from scipy.misc import imread
import cv2
import os
import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    #which_args = argparse.ArgumentParser()
    parser.add_argument('-f', '--gt', type=str, default='checkpoint/pose/gt.json', help='json of groundtruth')
    parser.add_argument('-o', '--output_img_dir', type=str, default='results/gt_vs_dt/', help='output directory ')
    args = parser.parse_args()
    return args

data_dir = '/home/ys/AE/ys-ae-pytorch/data/coco/'
ann_path = '/home/ys/AE/ys-ae-pytorch/data/coco/train/person_keypoints_train2014.json'
ref_dir = os.path.dirname(__file__)

flipRef = [i-1 for i in [1,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16] ]

part_labels = ['nose','eye_l','eye_r','ear_l','ear_r',
               'sho_l','sho_r','elb_l','elb_r','wri_l','wri_r',
               'hip_l','hip_r','kne_l','kne_r','ank_l','ank_r']
part_idx = {b:a for a, b in enumerate(part_labels)}    #part_idx={'eye_l': 1, 'nose': 0, 'elb_r': 8, 'sho_r': 6, 'wri_r': 10, 'sho_l': 5, 'hip_l': 11, 'wri_l': 9, 'ear_l': 3, 'hip_r': 12, 'ank_l': 15, 'ear_r': 4, 'eye_r': 2, 'elb_l': 7, 'ank_r': 16, 'kne_l': 13, 'kne_r': 14}

def image_path(idx):
    
    #path = '/home/ys/AE/ys-ae-pytorch/data/coco/train2014/COCO_train2014_' + str(idx).zfill(12) + '.jpg'
    path = '/home/ys/AE/pose-ae-train/results/test_imgs/'+str(idx)+'.jpg'
    return path

def load_image(idx):
    return imread(image_path(idx),mode='RGB')

def draw_limbs(inp, pred):  #pred=[x1,y1,1,x2,y2,1,...,x17,y17,1]
    def link(a, b, color):
        if part_idx[a] < pred.shape[0] and part_idx[b] < pred.shape[0]:
            a = pred[part_idx[a]]     #[x,y,1] 
            b = pred[part_idx[b]]     #[x,y,1]
            if a[2]>0.07 and b[2]>0.07:
                #cv2.line(inp, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, 1,cv2.LINE_AA)
                cv2.circle (inp,(int(a[0]), int(a[1])),2,(55,255,155),2)
                cv2.circle (inp,(int(b[0]), int(b[1])),2,(55,255,155),2)
    pred = np.array(pred).reshape(-1, 3)   #array([[x1,y1,1],[],[x17,y17,1]]]
    bbox = pred[pred[:,2]>0]
    #if bbox is not None:
       # a, b, c, d = bbox[:,0].min(), bbox[:,1].min(), bbox[:,0].max(), bbox[:,1].max()

   # cv2.rectangle(inp, (int(a), int(b)), (int(c), int(d)), (255, 255, 255), 1)

    link('nose', 'eye_l', (255, 0, 0))
    link('eye_l', 'eye_r', (255, 0, 0))
    link('eye_r', 'nose', (255, 0, 0))

    link('eye_l', 'ear_l', (255, 0, 0))
    link('eye_r', 'ear_r', (255, 0, 0))

    link('ear_l', 'sho_l', (255, 0, 0))
    link('ear_r', 'sho_r', (255, 0, 0))
    link('sho_l', 'sho_r', (255, 0, 0))
    link('sho_l', 'hip_l', (0, 255, 0))
    link('sho_r', 'hip_r',(0, 255, 0))
    link('hip_l', 'hip_r', (0, 255, 0))

    link('sho_l', 'elb_l', (0, 0, 255))
    link('elb_l', 'wri_l', (0, 0, 255))

    link('sho_r', 'elb_r', (0, 0, 255))
    link('elb_r', 'wri_r', (0, 0, 255))

    link('hip_l', 'kne_l', (255, 255, 0))
    link('kne_l', 'ank_l', (255, 255, 0))

    link('hip_r', 'kne_r', (255, 255, 0))
    link('kne_r', 'ank_r', (255, 255, 0))

def main():
    opt = parse_args()
    preds = []
 

    if opt.gt is not None:
        f=opt.gt
        print(f)
        with open(f,'r') as json_file:
            preds = json.loads(json_file.readline())

    last_id=0
    people_number=0
    for i in preds['annotations']:
        image_id=i['image_id']
        keypoints=i['keypoints']
        
        
        if image_id != last_id:
            if last_id !=0:
                img_last=cv2.imread(opt.output_img_dir+str(last_id)+'.jpg')
                text='gt_num:'+str(people_number)
                cv2.putText(img_last, text, (5,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), lineType=cv2.LINE_AA)
                cv2.imwrite(opt.output_img_dir+str(last_id)+'.jpg', img_last)
                people_number=0
            	
            img=load_image(image_id)
            draw_limbs(img, keypoints)
            print(image_id,i['image_id'])
            people_number+=1
        else:
            img=imread(opt.output_img_dir+str(image_id)+'.jpg',mode='RGB')
            draw_limbs(img, keypoints)
            print(image_id,i['id'])
            people_number +=1
        last_id=image_id
        cv2.imwrite(opt.output_img_dir+str(image_id)+'.jpg', img[:,:,::-1])
  
        
   

if __name__ == '__main__':
    main()