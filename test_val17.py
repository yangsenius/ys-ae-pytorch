import cv2
import torch
import tqdm
import os
import numpy as np
import pickle
from torch.nn import DataParallel
from data.ref import ref_dir, flipRef
from utils.misc import get_transform, kpt_affine, resize
from utils.group import HeatmapParser
from utils.misc import make_input, make_output

#valid_filepath = ref_dir + '/validation.pkl'
valid_filepath = '/home/ys/AE/ys-ae-pytorch/data/coco/annotations/person_keypoints_val2017.json'
valid_datapath = '/home/ys/AE/ys-ae-pytorch/data/coco/val2017/'
parser = HeatmapParser(detection_val=0.1)

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

def coco_eval(prefix, dt, gt):
    """
    Evaluate the result with COCO API
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    
    import json
    
    with open(prefix + '/dt.json', 'w') as f:
        json.dump(sum(dt, []), f)
 
    #coco = COCO(prefix + '/gt.json')
    coco = COCO(valid_filepath)
    
    coco_dets = coco.loadRes(prefix + '/dt.json')
    coco_eval = COCOeval(coco, coco_dets, "keypoints")
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

def genDtByPred(pred, image_id = 0):
    """
    Generate the json-style data for the output 
    """
    ans = []
    for i in pred:
        val = pred[i] if type(pred) == dict else i
        if val[:, 2].max()>0 :  # add: and float---- score
            tmp = {'image_id':int(image_id), "category_id": 1, "keypoints": [], "score":float(val[:, 2].mean())}
            p = val[val[:, 2]> 0][:, :2].mean(axis = 0)
            for j in val:
                if j[2]>0.:
                    tmp["keypoints"] += [float(j[0]), float(j[1]), 1]
                else:
                    tmp["keypoints"] += [float(p[0]), float(p[1]), 1]
            ans.append(tmp)
    return ans

def get_img(inp_res = 512):
    """
    Load validation images
    """
    '''if os.path.exists(valid_filepath) is False:
        from utils.build_valid import main
        main()

    dic = pickle.load(open(valid_filepath, 'rb'))
    paths, anns, idxes, info = [dic[i] for i in ['path', 'anns', 'idxes', 'info']]

    total = len(paths)
    tr = tqdm.tqdm( range(0, total), total = total )
    for i in tr:
        img = cv2.imread(paths[i])[:,:,::-1]
        yield anns[i], img'''
    f=valid_filepath
    import json
    with open(f,'r') as json_file:
            preds = json.loads(json_file.readline())
    total = len(preds['images'])
    val_imgs = tqdm.tqdm(range(0,total), total = total )
    for i in val_imgs:
        img=cv2.imread(valid_datapath+preds['images'][i]["file_name"])[:,:,::-1]
        image_id=preds['images'][i]['id']
        yield image_id,img
    
    '''for i in preds['images']:
        print(valid_datapath+i['file_name'])
        img=cv2.imread(valid_datapath+i['file_name'])[:,:,::-1]
        yield img'''

def test_func(model,**inputs):
    for i in inputs:
        inputs[i] = make_input(inputs[i])
    net = model.eval()

    forward_net=DataParallel(net.cuda())
    out = {}
    net = forward_net.eval()
    result = net(**inputs)
    if type(result)!=list and type(result)!=tuple:
        result = [result]
    out['preds'] = [make_output(i) for i in result]
    return out

def main():
    from main import options,Model_Checkpoints
    from models.posenet import PoseNet
    
    opts=options()
    mode=opts.mode
    model=PoseNet(nstack=opts.nstack,inp_dim=opts.inp_dim,oup_dim=opts.oup_dim)
    
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    epoch=Model_Checkpoints(opts).load_checkpoints(model,optimizer)
    print("Use the model which is trained by {} epoches".format(epoch))

    if opts.continue_exp is None:
        print("Warning: you must choose a trained model")
        
    
    def runner(imgs):
        return test_func(model, imgs=torch.Tensor(np.float32(imgs)))['preds']

    def do(image_id,img):
        ans, scores = multiperson(img, runner, mode)
        if len(ans) > 0:
            ans = ans[:,:,:3]

        pred = genDtByPred(ans,image_id)

        for i, score in zip( pred, scores ):
            i['score'] = float(score)
        return pred

    gts = []
    preds = []

    idx = 0
    for image_id,img in get_img(inp_res=-1):
        idx += 1
        
        preds.append(do(image_id,img))
    
    prefix = os.path.join('checkpoint', opts.continue_exp)
    coco_eval(prefix, preds, gts)

if __name__ == '__main__':
    main()