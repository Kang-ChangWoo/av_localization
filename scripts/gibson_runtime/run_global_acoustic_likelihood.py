#!/usr/bin/env python3
"""Global CUDA acoustic likelihood sweep for pre-rendered F3Loc RIRs."""
from __future__ import annotations

import argparse, csv, json, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from global_acoustic_likelihood_gpu import score_wall_distance_batch_cuda
from planar_acoustic_likelihood import PlanarRirConfig, map_xy_to_row_col, planar_rir_from_wall_distances, row_col_to_map_xy, wall_distances_from_candidate

ROOT=Path(__file__).resolve().parents[2]
DATA=Path(__import__('os').environ.get('AVFPLOC_DATA_ROOT','/file2/jeongeon/AV-FPLoc'))/'datasets/f3loc'
OBS=Path(__import__('os').environ.get('AVFPLOC_DATA_ROOT','/file2/jeongeon/AV-FPLoc'))/'rendered_rirs'
OUT=Path(__import__('os').environ.get('AVFPLOC_DATA_ROOT','/file2/jeongeon/AV-FPLoc'))/'reports/global_acoustic_likelihood'
CONDITIONS=('floorplan_closed','raw_scan_open')
POSES=(0,141,283)

def pose(path,i):
    return tuple(map(float,path.read_text().splitlines()[i].split()[:3]))

def global_states(map_array, desdf):
    h,w,o=desdf['desdf'].shape; t,l=int(desdf['t']),int(desdf['l'])
    rows=[]; cols=[]; xy=[]
    for r in range(h):
      for c in range(w):
        mr,mc=t+r*10,l+c*10
        if 0<=mr<map_array.shape[0] and 0<=mc<map_array.shape[1] and map_array[mr,mc]>=128:
          rows.append(r); cols.append(c); xy.append(row_col_to_map_xy(map_array.shape,(mr + 0.5, mc + 0.5)))
    return np.asarray(rows),np.asarray(cols),np.asarray(xy,dtype=np.float64)

def cache_distances(cache,map_array,rows,cols,xy,config):
    if cache.is_file():
      d=np.load(cache)
      if np.array_equal(d['rows'],rows) and np.array_equal(d['cols'],cols) and d['distances'].shape==(len(rows),config.ray_count): return d['distances']
    cache.parent.mkdir(parents=True,exist_ok=True); values=np.empty((len(rows),config.ray_count),np.float32)
    for i,p in enumerate(xy):
      values[i]=wall_distances_from_candidate(map_array,tuple(map(float,p)),config)
      if i%500==0: print(f'cached {i}/{len(rows)}',flush=True)
    np.savez_compressed(cache,rows=rows,cols=cols,distances=values); return values

def gt_state(map_array,desdf,p):
    mr,mc=map_xy_to_row_col(map_array.shape,p[:2]); r=int(np.rint((mr-int(desdf['t']))/10)); c=int(np.rint((mc-int(desdf['l']))/10)); o=desdf['desdf'].shape[2]; y=int(np.rint((p[2]%(2*np.pi))/(2*np.pi/o)))%o; return r,c,y

def summary(volume,rows,cols,xy,gt):
    r,c,y=gt; valid=np.isfinite(volume[rows,cols,:]); vals=volume[rows,cols,:]; gtval=volume[r,c,y]
    order=(vals.reshape(-1)>gtval).sum()+1
    spatial=np.nanmax(volume,axis=2); idx=np.nanargmax(spatial); br,bc=np.unravel_index(idx,spatial.shape)
    hit=np.where((rows==br)&(cols==bc))[0][0]; g=np.where((rows==r)&(cols==c))[0][0]
    return {'global_state_rank':int(order),'global_state_count':int(valid.sum()),'rank_percentile':float(order/valid.sum()),'best_x_m':float(xy[hit,0]),'best_y_m':float(xy[hit,1]),'best_yaw_index':int(np.nanargmax(volume[br,bc,:])),'best_spatial_error_m':float(np.linalg.norm(xy[hit]-xy[g]))}

def image(map_array,spatial,top,left,p,metrics,path,title):
    fig,ax=plt.subplots(figsize=(11,9)); ax.imshow(map_array,cmap='gray',vmin=0,vmax=255)
    finite=np.isfinite(spatial); rr,cc=np.where(finite); mr=int(0); # overlay DESDF grid using metadata-independent scatter passed elsewhere
    # The raw proxy scores are often compressed near one. Keep the raw array on
    # disk, but show per-map relative likelihood so blue/red remain legible.
    shown=np.full_like(spatial,np.nan,dtype=np.float32)
    values=spatial[finite]
    shown[finite]=(values-values.min())/max(float(values.max()-values.min()),1e-12)
    # Match the F3Loc likelihood visualization: cool=low, warm=high.
    im=ax.imshow(shown,cmap='coolwarm',vmin=0,vmax=1,alpha=.68,extent=(left,left+spatial.shape[1]*10,top+spatial.shape[0]*10,top),interpolation='nearest')
    gr,gc=map_xy_to_row_col(map_array.shape,p[:2]); br,bc=map_xy_to_row_col(map_array.shape,(metrics['best_x_m'],metrics['best_y_m']))
    # Keep the original F3Loc fixed-inch arrow convention so the poses remain
    # readable regardless of the map crop or resolution.
    arrow_style=dict(width=.2,scale_units='inches',units='inches',scale=1,headwidth=3,headlength=3,headaxislength=3,minlength=.1)
    ax.quiver(gc,gr,np.cos(p[2]),np.sin(p[2]),color='green',label='GT F3Loc pose',**arrow_style)
    ax.quiver(bc,br,np.cos(metrics['best_yaw_index']*2*np.pi/36),np.sin(metrics['best_yaw_index']*2*np.pi/36),color='blue',label='acoustic top-1',**arrow_style)
    ax.set_xlim(left,left+spatial.shape[1]*10); ax.set_ylim(top+spatial.shape[0]*10,top)
    ax.set_title(title); ax.legend(); fig.colorbar(im,ax=ax,label='relative yaw-max likelihood (per-map min-max)'); fig.tight_layout(); path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=150); plt.close(fig)

def waveform(observed,proxy,path,title):
    count=min(observed.shape[1],proxy.shape[1]); time=np.arange(count)/8000*1000
    fig,axes=plt.subplots(6,1,figsize=(11,10),sharex=True,constrained_layout=True)
    for channel,axis in enumerate(axes):
      observed_channel=observed[channel,:count]; proxy_channel=proxy[channel,:count]
      observed_scale=max(float(np.abs(observed_channel).max()),1e-12); proxy_scale=max(float(np.abs(proxy_channel).max()),1e-12)
      axis.plot(time,observed_channel/observed_scale,color='#2563eb',linewidth=.85,label='SoundSpaces 3D RIR' if channel==0 else None)
      axis.plot(time,proxy_channel/proxy_scale,color='#d97706',linewidth=.85,alpha=.9,label='2D GT proxy RIR' if channel==0 else None)
      axis.set_ylabel(f'ch{channel}')
      axis.grid(alpha=.2)
    axes[0].legend(loc='upper right'); axes[0].set_title(title+' (each trace independently peak-normalized)'); axes[-1].set_xlabel('Time (ms)')
    path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=150); plt.close(fig)

def main():
  a=argparse.ArgumentParser(); a.add_argument('--scene',default='Springhill'); a.add_argument('--batch-size',type=int,default=128); a.add_argument('--ray-count',type=int,default=72); a.add_argument('--output-root',type=Path,default=OUT); args=a.parse_args()
  if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
  config=PlanarRirConfig(ray_count=args.ray_count); base=args.output_root/args.scene; mp=np.asarray(Image.open(DATA/'gibson_t'/args.scene/'map.png').convert('L')); d=np.load(DATA/'desdf'/args.scene/'desdf.npy',allow_pickle=True).item(); rows,cols,xy=global_states(mp,d); distances=cache_distances(base/'wall_distances_72ray.npz',mp,rows,cols,xy,config)
  observations=[]; labels=[]; poses={i:pose(DATA/'gibson_t'/args.scene/'poses.txt',i) for i in POSES}
  for cond in CONDITIONS:
    for i in POSES:
      observations.append(np.load(OBS/args.scene/f'pose_{i:05d}'/cond/'rir_raw.npy')); labels.append((cond,i))
  scores=np.empty((len(labels),len(rows),36),np.float32); started=time.time()
  observed=torch.from_numpy(np.stack(observations))
  for start in range(0,len(rows),args.batch_size):
    end=min(start+args.batch_size,len(rows)); result=score_wall_distance_batch_cuda(torch.from_numpy(distances[start:end]),observed,config)
    scores[:,start:end]=result['likelihood'].detach().cpu().numpy()
    if start%1024==0: print(f'gpu {end}/{len(rows)}',flush=True)
  records=[]
  for oi,(cond,pi) in enumerate(labels):
    vol=np.full(d['desdf'].shape,np.nan,np.float32); vol[rows,cols,:]=scores[oi]; spatial=np.nanmax(vol,axis=2); gt=gt_state(mp,d,poses[pi]); m=summary(vol,rows,cols,xy,gt); root=base/cond/f'pose_{pi:05d}'; root.mkdir(parents=True,exist_ok=True); best_yaw=np.argmax(np.nan_to_num(vol,nan=-np.inf),axis=2).astype(np.int16); best_yaw[~np.isfinite(spatial)]=-1; np.save(root/'acoustic_likelihood_volume.npy',vol); np.save(root/'acoustic_likelihood_spatial.npy',spatial); np.save(root/'best_yaw_index.npy',best_yaw); image(mp,spatial,int(d['t']),int(d['l']),poses[pi],m,root/'global_likelihood.png',f'{cond} pose {pi}: global acoustic likelihood'); gt_hit=np.where((rows==gt[0])&(cols==gt[1]))[0][0]; gt_proxy=planar_rir_from_wall_distances(tuple(xy[gt_hit]),gt[2]*2*np.pi/36,distances[gt_hit],config); waveform(observations[oi],gt_proxy,root/'gt_proxy_vs_soundspaces_waveform.png',f'{cond} pose {pi}: GT-state 2D proxy vs 3D observation'); m.update({'condition':cond,'pose_index':pi,'artifacts':{'heatmap':str((root/'global_likelihood.png').relative_to(base)),'waveform':str((root/'gt_proxy_vs_soundspaces_waveform.png').relative_to(base))}}); records.append(m)
  metrics={'scene':args.scene,'device':torch.cuda.get_device_name(0),'batch_size':args.batch_size,'ray_count':args.ray_count,'valid_spatial_count':int(len(rows)),'state_count':int(len(rows)*36),'elapsed_seconds':time.time()-started,'records':records,'scaling':'raw exp(-mismatch) likelihood; no physical amplitude calibration or fusion'}
  (base/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');
  rows_html=''.join(f"<tr><td>{x['condition']}</td><td>{x['pose_index']}</td><td>{x['global_state_rank']}/{x['global_state_count']}</td><td>{x['rank_percentile']:.4f}</td><td>{x['best_spatial_error_m']:.3f}</td><td><a href='{x['artifacts']['heatmap']}'><img src='{x['artifacts']['heatmap']}'></a></td><td><a href='{x['artifacts']['waveform']}'><img src='{x['artifacts']['waveform']}'></a></td></tr>" for x in records)
  (base/'global_acoustic_likelihood_report.html').write_text(f"<!doctype html><meta charset=utf-8><style>body{{font-family:Arial;margin:28px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:8px;vertical-align:top}}img{{width:420px}}.n{{background:#fff3cd;padding:12px}}</style><h1>Global acoustic likelihood: {args.scene}</h1><p class=n>CUDA acoustic branch only. Values are raw normalized proxy similarity, not calibrated physical likelihood and not fused with F3Loc visual branch. Maps use blue=low, red=high relative likelihood; arrows are GT=green and acoustic top-1=blue. Waveform traces are independently peak-normalized per channel, for arrival-shape comparison only.</p><p>GPU: {metrics['device']} | states: {metrics['state_count']} | wall time: {metrics['elapsed_seconds']:.1f}s</p><table><tr><th>condition</th><th>pose</th><th>GT state rank</th><th>rank percentile</th><th>top-1 error m</th><th>global heatmap</th><th>GT 2D proxy vs 3D RIR</th></tr>{rows_html}</table>"); print(json.dumps(metrics))
if __name__=='__main__': main()
