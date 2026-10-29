"""Shared IO and geometry; no simulator, network or training side effects."""
from pathlib import Path
import hashlib, json, os
import numpy as np
from scipy.spatial.transform import Rotation

OPTICAL_TO_FRD = np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
TRANSLATIONS = np.array([[5,0,0],[0,0,0],[0,0,0],[0,0,-2],[0,0,2],[0,-5,0],[0,5,0]], float)
ROTATIONS = Rotation.from_euler('z', np.asarray([0,-15,15,0,0,0,0])[:,None], degrees=True).as_matrix()
VERBS = {1:('move forward',5,'units'),2:('turn left',15,'degrees'),3:('turn right',15,'degrees'),4:('ascend',2,'units'),5:('descend',2,'units'),6:('move left',5,'units'),7:('move right',5,'units')}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.%d.partial'%os.getpid())
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)

def recover(p,r):
    delta=np.einsum('nji,nj->ni',r[:-1],np.diff(p,axis=0))
    rel=np.einsum('nji,njk->nik',r[:-1],r[1:])
    distance=(((delta[:,None]-TRANSLATIONS[None])/[5.,5.,2.])**2).sum(2)
    trace=np.einsum('aij,nij->na',ROTATIONS,rel)
    angular=np.arccos(np.clip((trace-1)/2,-1,1))/(np.pi/12)
    costs=distance+angular**2
    return np.argmin(costs,axis=1).astype(int)+1, costs

def align(pred,target,scale=True):
    x,y=pred[:,:3,3],target[:,:3,3];mx,my=x.mean(0),y.mean(0)
    xc,yc=x-mx,y-my
    u,s,v=np.linalg.svd(yc.T@xc/len(x))
    d=np.array([1.,1.,np.linalg.det(u@v)]);r=u@np.diag(d)@v
    variance=float(np.mean(np.sum(xc*xc,axis=1)))
    factor=float(np.sum(s*d)/variance) if scale and variance>1e-12 else 1.
    t=my-factor*(r@mx)
    out=pred.copy();out[:,:3,:3]=r[None]@pred[:,:3,:3];out[:,:3,3]=factor*(x@r.T)+t
    return out,{'scale':factor,'R':r.tolist(),'t':t.tolist(),'translation_fit_rank':int(np.linalg.matrix_rank(xc)),'degenerate_translation':variance<=1e-12}

def metrics(pred,gt):
    delta=np.linalg.inv(gt[:-1])@gt[1:];dp=np.linalg.inv(pred[:-1])@pred[1:]
    e=np.linalg.inv(delta)@dp
    def stats(v):
        return {'rmse':float(np.sqrt(np.mean(v*v))),'median':float(np.median(v)),'p90':float(np.percentile(v,90)),'max':float(np.max(v))}
    return {'ate_m':stats(np.linalg.norm(pred[:,:3,3]-gt[:,:3,3],axis=1)),
            'orientation_deg':stats(np.degrees(Rotation.from_matrix(np.swapaxes(gt[:,:3,:3],1,2)@pred[:,:3,:3]).magnitude())),
            'rpe_step_translation_m':stats(np.linalg.norm(e[:,:3,3],axis=1)),
            'rpe_step_rotation_deg':stats(np.degrees(Rotation.from_matrix(e[:,:3,:3]).magnitude()))}

def segments(actions):
    out=[];t=0
    while t<len(actions):
        a=actions[t];length=1
        while t+length<len(actions) and actions[t+length]==a and length<3:length+=1
        label='The next action is stop.' if a==0 else 'The next action is %s %s %s.'%(VERBS[a][0],VERBS[a][1]*length,VERBS[a][2])
        out.append({'timestep':t,'action':label,'next_timestep':t+length if a else t,'primitive':a,'repeat':length});t+=length
    return out
