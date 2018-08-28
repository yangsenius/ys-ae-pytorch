import torch
import cv2
from scipy.misc import imread
import tqdm
import os
import numpy as np
import pickle
import json
import argparse

from data.ref import ref_dir, flipRef
from utils.misc import get_transform, kpt_affine, resize
from utils.group import HeatmapParser

flipRef = [i-1 for i in [1,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16] ]

part_labels = ['nose','eye_l','eye_r','ear_l','ear_r',
               'sho_l','sho_r','elb_l','elb_r','wri_l','wri_r',
               'hip_l','hip_r','kne_l','kne_r','ank_l','ank_r']
part_idx = {b:a for a, b in enumerate(part_labels)}  

parser = HeatmapParser(detection_val=0.1)

def options():
    opts=argparse.ArgumentParser()
    
    opts.add_argument('-c','--continue_exp',type=str, default=False, help='continue the experiment')
    opts.add_argument('-e','--exp',     type=str,          default='pose',help='begin a new experiment')
    opts.add_argument('-m', '--mode', type=str, default='single', help='scale mode')
    opts.add_argument('-img_dir', '--img_dir', type=str, default='test_imgs/', help='test_imgs')
    opts.add_argument('-o', '--output_image_path', type=str, default='/result/test_imgs', help='output image name')
    ##
    opts.add_argument('-nstack',         '--nstack',    type=int,  default=4,   help='the number of hourglass modules')
    opts.add_argument('-inp_dim',       '--inp_dim',    type=int,  default=256, help='the channels of the input for hourglass')
    opts.add_argument('-oup_dim',       '--oup_dim',    type=int,  default=68,  help='the channels of the output for prediction')
    opts.add_argument('-lr',                 '--lr',    type=float,default=2e-4,help='learning rate')
    opts.add_argument('-batchsize',   '--batchsize',    type=int,  default=16  ,help='batchsize')
    opts.add_argument('-input_res',   '--input_res',    type=int,  default=512 ,help='the resolution size of input')
    opts.add_argument('-output_res', '--output_res',    type=int,  default=128 ,help='the resolution size of output')
    opts.add_argument('-num_workers','--num_workers',   type=int,  default=2  , help='number of workers')
    opts.add_argument('-train_iters','--train_iters',   type=int,  default=1000,help='')
    opts.add_argument('-valid_iters','--valid_iters',   type=int,  default=10  ,help='')
    opts.add_argument('-max_num_people','--max_num_people',   type=int,  default=20,help='')
    opts.add_argument('-push_loss','--push_loss',   type=float,  default=1e-3,help='')
    opts.add_argument('-pull_loss','--pull_loss',   type=float,  default=1e-3,help='')
    opts.add_argument('-detection_loss','--detection_loss',   type=float,  default=1,help='')
    print("\n==================Options=================")
    from pprint import pprint
    pprint(vars(opts.parse_args()), indent=4)
    print("==========================================\n")
    return opts.parse_args()



def refine(det, tag, keypoints):
    """
    Given initial keypoint predictions, we identify missing joints
    """
    if len(tag.shape) == 3:
        tag = tag[:,:,:,None]

    tags = []
    for i in range(keypoints.shape[0]):
        if keypoints[i, 2] > 0:
            y, x = keypoints[i][:2].astype(np.int32)
            tags.append(tag[i, x, y])

    prev_tag = np.mean(tags, axis = 0)
    ans = []

    for i in range(keypoints.shape[0]):
        tmp = det[i, :, :]
        tt = (((tag[i, :, :] - prev_tag[None, None, :])**2).sum(axis = 2)**0.5 )
        tmp2 = tmp - np.round(tt)

        x, y = np.unravel_index( np.argmax(tmp2), tmp.shape )
        xx = x
        yy = y
        val = tmp[x, y]
        x += 0.5
        y += 0.5

        if tmp[xx, min(yy+1, det.shape[1]-1)]>tmp[xx, max(yy-1, 0)]:
            y+=0.25
        else:
            y-=0.25

        if tmp[min(xx+1, det.shape[0]-1), yy]>tmp[max(0, xx-1), yy]:
            x+=0.25
        else:
            x-=0.25

        x, y = np.array([y,x])
        ans.append((x, y, val))
    ans = np.array(ans)

    if ans is not None:
        for i in range(17):
            if ans[i, 2]>0 and keypoints[i, 2]==0:
                keypoints[i, :2] = ans[i, :2]
                keypoints[i, 2] = 1 

    return keypoints

def multiperson(img, func, mode):
    """
    1. Resize the image to different scales and pass each scale through the network
    2. Merge the outputs across scales and find people by HeatmapParser
    3. Find the missing joints of the people with a second pass of the heatmaps
    """
    if mode == 'multi':
        scales = [2, 1., 0.5]
    else:
        scales = [1]

    height, width = img.shape[0:2]
    center = (width/2, height/2)
    dets, tags = None, []
    for idx, i in enumerate(scales):
        scale = max(height, width)/200
        input_res = max(height, width)
        inp_res = int((i * 512 + 63)//64 * 64)
        res = (inp_res, inp_res)

        mat_ = get_transform(center, scale, res)[:2]
        inp = cv2.warpAffine(img, mat_, res)/255

        def array2dict(tmp):
            return {
                'det': tmp[0][:,:,:17],
                'tag': tmp[0][:,-1, 17:34]
            }
        prexx1=func([inp])
        prexx2=func([inp[:,::-1]])
        tmp1 = array2dict(func([inp]))
        tmp2 = array2dict(func([inp[:,::-1]]))

        tmp = {}
        for ii in tmp1:
            tmp[ii] = np.concatenate((tmp1[ii], tmp2[ii]),axis=0)

        det = tmp['det'][0, -1] + tmp['det'][1, -1, :, :, ::-1][flipRef]
        if det.max() > 10:
            continue
        if dets is None:
            dets = det
            mat = np.linalg.pinv(np.array(mat_).tolist() + [[0,0,1]])[:2]
        else:
            dets = dets + resize(det, dets.shape[1:3]) 

        if abs(i-1)<0.5:
            res = dets.shape[1:3]
            tags += [resize(tmp['tag'][0], res), resize(tmp['tag'][1,:, :, ::-1][flipRef], res)]

    if dets is None or len(tags) == 0:
        return [], []

    tags = np.concatenate([i[:,:,:,None] for i in tags], axis=3)
    dets = dets/len(scales)/2
    
    dets = np.minimum(dets, 1)
    grouped = parser.parse(np.float32([dets]), np.float32([tags]))[0]


    scores = [i[:, 2].mean() for  i in grouped]

    for i in range(len(grouped)):
        grouped[i] = refine(dets, tags, grouped[i])

    if len(grouped) > 0:
        grouped[:,:,:2] = kpt_affine(grouped[:,:,:2] * 4, mat)
    return grouped, scores
    
def draw_limbs(inp, pred):  #pred=[x1,y1,1,x2,y2,1,...,x17,y17,1]
    def link(a, b, color):
        if part_idx[a] < pred.shape[0] and part_idx[b] < pred.shape[0]:
            
            a = pred[part_idx[a]]     #[x,y,1] 
            b = pred[part_idx[b]]     #[x,y,1]
            
            if a[2]>0.07 and b[2]>0.07:
                cv2.line(inp, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, 1,cv2.LINE_AA)

    pred = np.array(pred).reshape(-1, 3)   #array([[x1,y1,1],[],[x17,y17,1]]]
    bbox = pred[pred[:,2]>0]
    a, b, c, d = bbox[:,0].min(), bbox[:,1].min(), bbox[:,0].max(), bbox[:,1].max()
    
    
 
   
    #cv2.rectangle(inp, (int(a), int(b)), (int(c), int(d)), (255, 255, 255), 1)

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

def genDtByPred(pred, image_id = 0):
    """
    Generate the json-style data for the output 
    """
    ans = []
    for i in pred:
        val = pred[i] if type(pred) == dict else i
        if val[:, 2].max()>0 and float(val[:, 2].mean())>0.15 :
            tmp = {'image_id':int(image_id), "category_id": 1, "keypoints": [], "score":float(val[:, 2].mean())}
            p = val[val[:, 2]> 0][:, :2].mean(axis = 0)
            for j in val:
                if j[2]>0.:
                    tmp["keypoints"] += [float(j[0]), float(j[1]), 1]
                else:
                    tmp["keypoints"] += [float(p[0]), float(p[1]), 1]
            ans.append(tmp)
    return ans


def main():
    from main import Model_Checkpoints
    from test import test_func
    from models.posenet import PoseNet

    opts = options()
    mode = opts.mode
 
    model = PoseNet(nstack=opts.nstack,inp_dim=opts.inp_dim,oup_dim=opts.oup_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    epoch=Model_Checkpoints(opts).load_checkpoints(model,optimizer)
    print("Use the model which is trained by {} epoches".format(epoch))
    def runner(imgs):
        return test_func(model, imgs=torch.Tensor(np.float32(imgs)))['preds']

    def do(img):
        ans, scores = multiperson(img, runner, mode)
        if len(ans) > 0:
            ans = ans[:,:,:3]
       # print(ans)
       # print(scores)
        pred = genDtByPred(ans)

        for i, score in zip( pred, scores ):
            i['score'] = float(score)
        return pred

    gts = []
    preds = []
    prefix = os.path.join('checkpoint', opts.continue_exp)
    idx = 0
    img_dir=opts.img_dir
    last_id=1000000
    if img_dir:
        f_list = os.listdir(img_dir)
            #resume_file={'resume_file':i  for i in f_list if os.path.splitext(i)[-1] == '.tar'}['resume_file']
        for img_name in f_list:
            print('xx')
            img=cv2.imread(img_dir+img_name)[:,:,::-1]
            cv2.imwrite('pose_results/'+img_name, img[:,:,::-1])
            preds=do(img)
           
            
            #with open(prefix + '/img_dt.json', 'wb') as f:
                #json.dump(sum([preds], []), f)
            
            for i in preds:
                
                keypoints=i['keypoints']
                
            #if i['score']<0.15:
            #    continue
                img=imread('pose_results/'+img_name,mode='RGB')
                draw_limbs(img, keypoints)

                cv2.imwrite('pose_results/'+img_name, img[:,:,::-1])
                print("{} has been estimated".format(img_name))
    
    

if __name__ == '__main__':
    main()
