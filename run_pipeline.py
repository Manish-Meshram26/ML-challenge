"""Optimized business entity resolution. Final submissions are written to output/.

Training uses sampled reference entities, but scans ALL source 2/3 distractors.
Validation keeps all retrieved negatives and includes unretrieved truths in scoring.
"""
from pathlib import Path
import sys,os,csv,re,json,time,math,gc,pickle,argparse,zlib,unicodedata,hashlib,shutil
from collections import defaultdict,Counter
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'.deps'))
import numpy as np
from rapidfuzz import fuzz,process
from rapidfuzz.distance import JaroWinkler,Levenshtein
from text_unidecode import unidecode
RESULTS=P/'output'; RESULTS.mkdir(parents=True,exist_ok=True)
OUT=RESULTS/'.cache'; OUT.mkdir(parents=True,exist_ok=True)
MODELS=P/'models'; MODELS.mkdir(parents=True,exist_ok=True)
VERSION=1
START=time.monotonic()
def log(s): print(f'[{(time.monotonic()-START)/60:.1f}m] {s}',flush=True)
def probabilities(model,x):
 return model.booster_.predict(x,num_threads=model.n_jobs)

def check():
 from utils.validate_submission import validate_streaming
 validate_streaming()
def commit_output(temp,path):
 for _ in range(8):
  try: temp.replace(path); return
  except PermissionError: time.sleep(.25)
 # Some Windows file watchers deny rename even after writers close.
 shutil.copyfile(temp,path)
def rows(split,source):
 with (P/f'dataset/{split}/{split}_source{source}.tsv').open(encoding='utf-8',newline='') as f:
  yield from csv.DictReader(f,delimiter='\t')
def normalized_rows(split,source,country):
 """Cache normalized source rows once; replay bounded chunks for reference shards."""
 path=OUT/f'cache_{split}_{country}_{source}.pkl'; stamp=path.with_suffix('.json')
 raw_path=P/f'dataset/{split}/{split}_source{source}.tsv'
 signature={'version':VERSION,'size':raw_path.stat().st_size,'mtime':raw_path.stat().st_mtime_ns}
 if path.exists() and stamp.exists() and json.loads(stamp.read_text())==signature:
  with path.open('rb') as f:
   while True:
    try: chunk=pickle.load(f)
    except EOFError: break
    yield from chunk
  return
 if stamp.exists(): stamp.unlink()
 chunk=[]
 with path.open('wb') as f:
  for raw in rows(split,source):
   if raw['country'].lower().strip()!=country: continue
   r=record(raw); chunk.append(r); yield r
   if len(chunk)>=25000: pickle.dump(chunk,f,pickle.HIGHEST_PROTOCOL); chunk=[]
  if chunk: pickle.dump(chunk,f,pickle.HIGHEST_PROTOCOL)
 stamp.write_text(json.dumps(signature),encoding='utf-8')
def stable(s): return zlib.crc32(s.encode('utf-8'))
WORDS=re.compile(r'[^\w]+',re.UNICODE)
URL=re.compile(r'(?:https?://)?(?:www\.)?([a-z0-9-]+)\.(?:com|net|org|in|fr|co)(?:\.[a-z]+)?(?:/\S*)?')
NUM=re.compile(r'\d+')
ABBR={'pvt':'private','ltd':'limited','inc':'incorporated','corp':'corporation','co':'company',
 'intl':'international','mfg':'manufacturing','svcs':'services','mgmt':'management',
 'rd':'road','ave':'avenue','blvd':'boulevard','hwy':'highway','ste':'suite'}
LEGAL=set('private limited incorporated corporation company llc llp pte pvt ltd inc corp co sa sas sarl eurl and the'.split())
def clean(s):
 s=URL.sub(r' \1 ',s.lower()).replace('&',' and ')
 if not s.isascii(): s=unidecode(s)
 return ' '.join(ABBR.get(w,str(int(w)) if w.isdigit() and len(w)<12 else w) for w in WORDS.sub(' ',s).split())
def record(r):
 n,a=clean(r['business_name']),clean(r['business_address'])
 core=' '.join(w for w in n.split() if w not in LEGAL)
 return (r['entity_id'],n,a,r['country'].lower().strip(),core or n)
def keys(r):
 _,n,a,c,core=r; nt=set(core.split()); at=a.split(); ns=NUM.findall(a)
 out=set()
 for tag,s in [('n',n),('c',core),('s',' '.join(sorted(nt))),('a',a)]:
  if s: out.add(tag+':'+s)
 if len(core)>=5:
  out.add('p:'+core[:6]); out.add('q:'+core[-6:])
 for w in nt:
  if len(w)>=3: out.add('t:'+w)
  if len(w)>=6:
   out.add('w:'+w[:4]+w[-2:])
 # Combinations of common words are often distinctive even when each word is not.
 ts=sorted(nt)
 for i,u in enumerate(ts[:8]):
  for v in ts[i+1:8]:
   out.add('j:'+u+':'+v)
   if len(u)>=4 and len(v)>=4: out.add('k:'+u[:4]+':'+v[:4])
 compact=core.replace(' ','')
 if compact:
  out.add('m:'+compact)
  if len(compact)>=7: out.add('f:'+compact[:8])
 if len(at)>=2: out.add('h:'+' '.join(at[:3]))
 for w in set(at):
  if len(w)>=5 and not w.isdigit(): out.add('b:'+w)
 for u,v in zip(at,at[1:]):
  if len(u)>=3 and len(v)>=3 and not u.isdigit() and not v.isdigit(): out.add('e:'+u+':'+v)
 # Numeric+word address keys survive reordered addresses and name translation.
 if ns:
  for number in set(ns[:3]):
   for w in set(at):
    if len(w)>=4 and not w.isdigit(): out.add('d:'+number+':'+w)
  nums=sorted(set(ns))[:6]
  for i,u in enumerate(nums):
   for v in nums[i+1:]: out.add('v:'+u+':'+v)
  for u in ts[:4]:
   if len(u)>=3:
    for n in nums: out.add('x:'+u[:4]+':'+n)
 return out
CAP={'n':150,'c':150,'s':150,'a':100,'p':60,'q':40,'t':45,'w':35,'h':100,'d':45,'b':15,'e':35,
 'j':100,'k':75,'m':150,'f':80,'v':60,'x':60}
class Index:
 def __init__(self,records):
  self.records=records; self.inv={}; self.context_counts=Counter(k for r in records for k in context_keys(r))
  for i,r in enumerate(records):
   for k in keys(r):
    if k not in self.inv: self.inv[k]=i
    else:
     v=self.inv[k]
     if v is None: continue
     if isinstance(v,int): self.inv[k]=[v,i]
     elif len(v)<CAP[k[0]]: v.append(i)
     else: self.inv[k]=None
  self.inv={k:v for k,v in self.inv.items() if v is not None}
 def restrict_global(self,split,country):
  # Use full-reference block frequencies even when fitting on sampled anchors.
  counts=Counter(); wanted=set(self.inv)|set(self.context_counts)
  for raw in rows(split,1):
   if raw['country'].lower().strip()!=country: continue
   for k in keys(record(raw)):
    if k in wanted: counts[k]+=1
  self.inv={k:v for k,v in self.inv.items() if counts[k]<=CAP[k[0]]}
  self.context_counts={k:counts[k] for k in self.context_counts}
 def query(self,r):
  hits=set()
  for k in keys(r):
   v=self.inv.get(k)
   if v is not None:
    if isinstance(v,int): hits.add(v)
    else: hits.update(v)
  return hits
def overlap(a,b):
 u=a|b
 return len(a&b)/len(u) if u else 0.
FEATURES=[]
for f in ['name','core','address']:
 FEATURES += [f+'_'+x for x in ['ratio','sort','set','partial','jw','jaccard','contain','length_ratio','exact','missing']]
FEATURES += ['number_jaccard','number_conflict','number_common','number_missing',
 'first_number_equal','name_number_conflict','name_length','address_length','source3','name_addr_product',
 'compact_name_ratio','compact_name_partial']
def features(x,y):
 out=[]
 for a,b in [(x[1],y[1]),(x[4],y[4]),(x[2],y[2])]:
  sa,sb=set(a.split()),set(b.split()); both=bool(a and b)
  out.extend([fuzz.ratio(a,b)/100 if both else 0,
   fuzz.token_sort_ratio(a,b)/100 if both else 0,
   fuzz.token_set_ratio(a,b)/100 if both else 0,
   fuzz.partial_ratio(a,b)/100 if both else 0,
   JaroWinkler.normalized_similarity(a,b) if both else 0,
   overlap(sa,sb),len(sa&sb)/max(1,min(len(sa),len(sb))),
   min(len(a),len(b))/max(1,len(a),len(b)),float(both and a==b),float(not both)])
 na,nb=set(NUM.findall(x[2])),set(NUM.findall(y[2]))
 ia,ib=NUM.findall(x[2]),NUM.findall(y[2]); nn1,nn2=set(NUM.findall(x[1])),set(NUM.findall(y[1]))
 out.extend([overlap(na,nb),float(bool(na and nb) and not na&nb),len(na&nb),float(not na or not nb),
  float(bool(ia and ib) and ia[0]==ib[0]),float(bool(nn1 and nn2) and nn1!=nn2),
  min(len(x[1]),len(y[1])),min(len(x[2]),len(y[2])),float(y[0].startswith('S3')),out[10]*out[20],
  fuzz.ratio(x[4].replace(' ',''),y[4].replace(' ',''))/100 if x[4] and y[4] else 0,
  fuzz.partial_ratio(x[4].replace(' ',''),y[4].replace(' ',''))/100 if x[4] and y[4] else 0])
 return out
CONTEXT_FEATURES=['reference_name_frequency','reference_core_frequency','reference_address_frequency']
SOURCE_FEATURES=['source_core_frequency','source_core_address_frequency']
EXTRA_FEATURES=['unmatched_left_numbers','unmatched_right_numbers','number_delta_log','number_near_miss',
 'unmatched_number_similarity','name_substitution_rate','name_insertion_rate','name_deletion_rate',
 'worst_name_token_similarity','mean_name_token_similarity']
def extra_features(x,y):
 na,nb=set(NUM.findall(x[2])),set(NUM.findall(y[2])); da,db=na-nb,nb-na
 ds=[abs(int(a)-int(b)) for a in da for b in db if len(a)<12 and len(b)<12]
 delta=min(ds) if ds else 0
 out=[len(da),len(db),math.log1p(min(delta,1000000)),float(bool(ds) and delta<=2),
  max((fuzz.ratio(a,b)/100 for a in da for b in db),default=0)]
 a,b=x[4].replace(' ',''),y[4].replace(' ',''); counts=Counter(op.tag for op in Levenshtein.editops(a,b))
 out.extend(counts[k]/max(1,len(a),len(b)) for k in ['replace','insert','delete'])
 aa,bb=x[4].split(),y[4].split()
 sims=[max((fuzz.ratio(u,v)/100 for v in bb),default=0) for u in aa]
 out.extend([min(sims,default=0),sum(sims)/max(1,len(sims))])
 return out
FAST_COLS=[0,10,11,12,20,21,22,19,29,42,43,44,45,46]
def source_profile(split,country):
 path=OUT/f'profile_{split}_{country}.pkl'
 done=path.with_suffix('.done')
 if path.exists() and done.exists():
  with path.open('rb') as f: return pickle.load(f)
 names=Counter(); joint=Counter()
 for source in [2,3]:
  for r in normalized_rows(split,source,country):
   names[r[4]]+=1; joint[(r[4],r[2])]+=1
 with path.open('wb') as f: pickle.dump((names,joint),f,pickle.HIGHEST_PROTOCOL)
 done.write_text(str(VERSION),encoding='utf-8')
 return names,joint
def source_context(r,profile):
 return [math.log1p(min(1000,profile[0][r[4]])),math.log1p(min(1000,profile[1][(r[4],r[2])]))]
def fast_features(x,y,context):
 n=bool(x[1] and y[1]); c=bool(x[4] and y[4]); a=bool(x[2] and y[2])
 return [fuzz.ratio(x[1],y[1])/100 if n else 0,
  fuzz.ratio(x[4],y[4])/100 if c else 0,
  fuzz.token_sort_ratio(x[4],y[4])/100 if c else 0,
  fuzz.token_set_ratio(x[4],y[4])/100 if c else 0,
  fuzz.ratio(x[2],y[2])/100 if a else 0,
  fuzz.token_sort_ratio(x[2],y[2])/100 if a else 0,
  fuzz.token_set_ratio(x[2],y[2])/100 if a else 0,float(not c),float(not a),*context]
def fast_matrix(pairs,recs,context,threads):
 """Batch C++ similarity kernels across CPU cores, preserving scalar semantics."""
 n=len(pairs); width=9+len(context[0])+len(pairs[0][2]); out=np.empty((n,width),dtype=np.float32); gate_mask=np.zeros(n,dtype=bool)
 for field,columns,scorers in [(1,[0],[fuzz.ratio]),(4,[1,2,3],[fuzz.ratio,fuzz.token_sort_ratio,fuzz.token_set_ratio]),
  (2,[4,5,6],[fuzz.ratio,fuzz.token_sort_ratio,fuzz.token_set_ratio])]:
  left=[recs[j][field] for j,r,sc in pairs]; right=[r[field] for j,r,sc in pairs]
  missing=np.array([not a or not b for a,b in zip(left,right)])
  for col,scorer in zip(columns,scorers):
   raw=process.cpdist(left,right,scorer=scorer,dtype=np.float64,workers=threads).ravel()
   out[:,col]=raw/100
   if col in [1,2]: gate_mask|=raw>=45
   if col==3: gate_mask|=(raw>=85)&~missing
   if col==5: gate_mask|=(raw>=55)&~missing
   if col==6: gate_mask|=(raw>=85)&~missing
   out[missing,col]=0
  if field==4: out[:,7]=missing
  if field==2: out[:,8]=missing
 out[:,9:]=np.array([context[j]+sc for j,r,sc in pairs],dtype=np.float32)
 return out,gate_mask
def context_keys(r):
 return ('n:'+r[1],'c:'+r[4],'a:'+r[2])
def context_features(r,counts):
 return [math.log1p(min(1000,counts.get(k,0))) if len(k)>2 else 0. for k in context_keys(r)]
def gate(x,y):
 # Broad inexpensive filter, part of measured retrieval (never silently ignored).
 return bool(max(fuzz.ratio(x[4],y[4]),fuzz.token_sort_ratio(x[4],y[4]))>=45
  or (x[4] and y[4] and fuzz.token_set_ratio(x[4],y[4])>=85)
  or (x[2] and y[2] and (fuzz.token_sort_ratio(x[2],y[2])>=55 or fuzz.token_set_ratio(x[2],y[2])>=85)))
def truth(ids):
 out={i:set() for i in ids}; seen=set()
 with (P/'dataset/train/train_ground_truth.tsv').open(encoding='utf-8',newline='') as f:
  for r in csv.DictReader(f,delimiter='\t'):
   k=r['source1_entity_id']
   if k in out:
    seen.add(k); out[k]={x.strip() for x in r['matched_entity_ids'].split(',') if x.strip()}
 if len(seen)!=len(out): raise ValueError(f'Missing ground-truth rows: {len(out)-len(seen)}')
 return out
def group(r):
 # Exact duplicate reference records stay together; IDs are never model inputs.
 return stable(r[3]+'|'+r[1]+'|'+r[2])%10
def prepare(args):
 if (OUT/'prepare_complete.json').exists(): (OUT/'prepare_complete.json').unlink()
 selected=[]; counts=Counter()
 for raw in rows('train',1):
  counts[raw['country']]+=1
  if stable(raw['entity_id'])%1000000 < args.sample_rate*1000000: selected.append(record(raw))
 log(f'Selected {len(selected)} reference entities from {dict(counts)}')
 gt=truth({r[0] for r in selected}); idpos={r[0]:i for i,r in enumerate(selected)}
 groups=np.array([group(r) for r in selected],dtype=np.uint8)
 manifest={'version':VERSION,'features':FEATURES,'sample_rate':args.sample_rate,'records':selected,
  'truth':{k:sorted(v) for k,v in gt.items()},'groups':groups.tolist()}
 (OUT/'training_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
 xs=[]; meta=[]; total=0; retrieved=0; forced=0
 with (OUT/'features.f32').open('wb') as xf,(OUT/'pairs.tsv').open('w',encoding='utf-8',newline='') as pf:
  writer=csv.writer(pf,delimiter='\t'); writer.writerow(['anchor','candidate','label','retrieved'])
  def flush():
   nonlocal total
   if xs:
    np.asarray(xs,dtype=np.float32).tofile(xf); writer.writerows(meta)
    total+=len(xs); xs.clear(); meta.clear(); xf.flush(); pf.flush()
  for country in sorted({r[3] for r in selected}):
   recs=[r for r in selected if r[3]==country]; idx=Index(recs)
   idx.restrict_global('train',country)
   log(f'{country}: {len(recs)} anchors, {len(idx.inv)} eligible keys')
   # Positive injection is ONLY for fitting entities; validation/audit never use GT retrieval.
   inject=defaultdict(list)
   for j,r in enumerate(recs):
    if groups[idpos[r[0]]]<6:
     for k in gt[r[0]]: inject[k].append(j)
   for source in [2,3]:
    n=0
    for raw in rows('train',source):
     if raw['country'].lower().strip()!=country: continue
     n+=1; r=record(raw); found={j for j in idx.query(r) if gate(recs[j],r)}
     cand=found|set(inject.get(r[0],[]))
     for j in cand:
      ref=recs[j]; lab=int(r[0] in gt[ref[0]]); ret=int(j in found)
      retrieved+=lab*ret; forced+=lab*(1-ret)
      xs.append(features(ref,r)); meta.append([idpos[ref[0]],r[0],lab,ret])
     if len(xs)>=20000: flush()
     if n%500000==0: log(f'{country} S{source}: {n:,} scanned, {total+len(xs):,} pairs')
    flush(); log(f'{country} S{source} complete')
   del idx; gc.collect()
 log(f'Prepared {total:,} pairs; retrieved positives {retrieved:,}; fit-only injected positives {forced:,}')
 (OUT/'prepare_complete.json').write_text(json.dumps({'pairs':total,'seconds':time.monotonic()-START,'version':VERSION}),encoding='utf-8')
def score_arrays(anchor,label,prob,threshold,truth_count,universe,retrieved=None):
 mask=prob>=threshold
 if retrieved is not None: mask &= retrieved
 pred=np.bincount(anchor[mask],minlength=len(truth_count))
 tp=np.bincount(anchor[mask],weights=label[mask],minlength=len(truth_count))
 den=pred+0.25*truth_count
 scores=np.divide(1.25*tp,den,out=np.ones(len(den)),where=den!=0)
 return float(scores[universe].mean()),scores,pred,tp
def fit(args):
 import pandas as pd
 import lightgbm as lgb
 m=json.loads((OUT/'training_manifest.json').read_text(encoding='utf-8'))
 complete=json.loads((OUT/'prepare_complete.json').read_text())
 # Rename original ratio column labels; numeric feature positions are unchanged.
 for i in [7,17,27]:
  if m['features'][i]==FEATURES[i].replace('_ratio',''): m['features'][i]=FEATURES[i]
 if m['version']!=VERSION or m['features']!=FEATURES: raise ValueError('Stale feature cache; rerun prepare')
 meta=pd.read_csv(OUT/'pairs.tsv',sep='\t',dtype={'anchor':'int32','label':'uint8','retrieved':'uint8'})
 if meta.duplicated(['anchor','candidate']).any(): raise ValueError('Duplicate candidate pairs would invalidate set-based scoring')
 if len(meta)!=complete['pairs'] or (OUT/'features.f32').stat().st_size!=len(meta)*len(FEATURES)*4:
  raise ValueError('Incomplete or mismatched prepared features')
 base=np.memmap(OUT/'features.f32',dtype='float32',mode='r',shape=(len(meta),len(FEATURES)))
 a=meta.anchor.to_numpy(); y=meta.label.to_numpy(); ret=meta.retrieved.to_numpy().astype(bool)
 log('Measuring reference ambiguity from all training reference records')
 wanted={r[3]+'|'+k for r in m['records'] for k in context_keys(r)}; counts=Counter()
 for raw in rows('train',1):
  r=record(raw)
  for k in context_keys(r):
   key=r[3]+'|'+k
   if key in wanted: counts[key]+=1
 context=np.array([[math.log1p(min(1000,counts[r[3]+'|'+k])) if len(k)>2 else 0. for k in context_keys(r)] for r in m['records']],dtype=np.float32)
 del wanted,counts
 log('Measuring source-record corroboration frequencies (no labels used)')
 wanted=set(meta.candidate); source_stats={}; source_records={}
 for country in sorted({r[3] for r in m['records']}):
  profile=source_profile('train',country)
  for source in [2,3]:
   for r in normalized_rows('train',source,country):
    if r[0] in wanted: source_stats[r[0]]=source_context(r,profile); source_records[r[0]]=r
  del profile; gc.collect()
 sc=np.array([source_stats[k] for k in meta.candidate],dtype=np.float32)
 log('Building number-contradiction and character-edit features')
 extra=np.empty((len(meta),len(EXTRA_FEATURES)),dtype=np.float32)
 for i,(j,k) in enumerate(zip(a,meta.candidate)):
  extra[i]=extra_features(m['records'][j],source_records[k])
  if i and i%500000==0: log(f'Additional features: {i:,}/{len(meta):,}')
 X=np.column_stack((base,context[a],sc,extra)); del base,wanted,source_stats,source_records,sc,extra
 feature_names=FEATURES+CONTEXT_FEATURES+SOURCE_FEATURES+EXTRA_FEATURES
 g=np.array(m['groups']); ng=g[a]; tc=np.array([len(m['truth'][r[0]]) for r in m['records']])
 train=ng<6; tune=(ng>=6)&(ng<8); audit=ng>=8
 params=dict(n_estimators=900,learning_rate=.045,num_leaves=47,min_child_samples=50,
  reg_lambda=5.,reg_alpha=.2,colsample_bytree=.9,subsample=.85,subsample_freq=1,
  n_jobs=args.threads,verbosity=-1,random_state=42)
 log(f'Fit {train.sum():,}; tune {tune.sum():,}; audit {audit.sum():,}; positives {y[train].sum():,}')
 model=lgb.LGBMClassifier(**params)
 model.fit(X[train],y[train],eval_set=[(X[tune],y[tune])],
  callbacks=[lgb.early_stopping(60),lgb.log_evaluation(100)],feature_name=feature_names)
 prob=np.zeros(len(y),dtype=np.float32)
 for start in range(0,len(y),100000): prob[start:start+100000]=probabilities(model,X[start:start+100000])
 np.save(OUT/'validation_probabilities.npy',prob)
 fast=lgb.LGBMClassifier(**{**params,'n_estimators':250,'num_leaves':31})
 fast.fit(X[train][:,FAST_COLS],y[train])
 fast_prob=np.zeros(len(y),dtype=np.float32)
 for start in range(0,len(y),100000): fast_prob[start:start+100000]=probabilities(fast,X[start:start+100000,FAST_COLS])
 # Use the tuning set only: lose at most 0.05% of retrieved true pairs to this speed filter.
 positive_scores=fast_prob[tune&ret&(y==1)]
 fast_threshold=float(min(.01,np.quantile(positive_scores,.0005))) if len(positive_scores) else 0.
 fast_keep=fast_prob>=fast_threshold
 prob[~fast_keep]=0.
 np.save(OUT/'validation_probabilities.npy',prob)
 thresholds=np.unique(np.r_[np.linspace(.01,.99,99),.995,.999,.9995])
 tune_ids=(g>=6)&(g<8); audit_ids=g>=8
 best=(-1,None)
 for t in thresholds:
  sc=score_arrays(a,y,prob,t,tc,tune_ids,ret)[0]
  if sc>best[0]: best=(sc,float(t))
 threshold=best[1]
 countries=np.array([r[3] for r in m['records']]); country_thresholds={}; regional_gain=0.
 for c in sorted(set(countries)):
  universe=tune_ids&(countries==c)
  values=[(score_arrays(a,y,prob,t,tc,universe,ret)[0],float(t)) for t in thresholds]
  cs,ct=max(values,key=lambda v:v[0]); country_thresholds[c]=ct
  regional_gain+=cs*int(universe.sum())/int(tune_ids.sum())
 if regional_gain-best[0]<.001: country_thresholds={}
 decision=np.array([country_thresholds.get(c,threshold) for c in countries])[a]
 # Select the screening cutoff by end-to-end tuning score, not retrieval alone.
 # Prefer fewer expensive comparisons among cutoffs within 0.0002 of the best score.
 filter_options=np.unique(np.r_[fast_threshold,.0003,.001,.003,.01,.03,.05,.1,.2,.3,.4,.5])
 filter_results=[]
 for ft in filter_options[filter_options>=fast_threshold]:
  value=score_arrays(a,y,prob,decision,tc,tune_ids,ret&(fast_prob>=ft))[0]
  filter_results.append((float(ft),value))
 best_filter_score=max(v for _,v in filter_results)
 fast_threshold=max(ft for ft,v in filter_results if v>=best_filter_score-.0002)
 fast_keep=fast_prob>=fast_threshold; prob[~fast_keep]=0.
 np.save(OUT/'validation_probabilities.npy',prob)
 np.save(OUT/'fast_probabilities.npy',fast_prob)
 sc,scores,pred,tp=score_arrays(a,y,prob,decision,tc,audit_ids,ret)
 oracle=score_arrays(a,y,y.astype(float),.5,tc,audit_ids,ret)[0]
 audit_recall=float((y[audit]*ret[audit]).sum()/max(1,tc[audit_ids].sum()))
 report={'threshold':threshold,'tuning_macro_f05':best[0],'audit_macro_f05':sc,
  'country_thresholds':country_thresholds,'regional_tuning_gain':regional_gain-best[0],
  'selected_tuning_macro_f05':score_arrays(a,y,prob,decision,tc,tune_ids,ret)[0],
  'filter_tuning_results':filter_results,
  'audit_blocking_recall':audit_recall,'audit_oracle_macro_f05':oracle,
  'best_iteration':model.best_iteration_,'reference_sample':len(g),'pair_count':len(y),
  'fast_threshold':fast_threshold,'fast_candidate_retention':float(fast_keep[ret].mean()),
  'audit_fast_positive_retention':float(fast_keep[audit&ret&(y==1)].mean()),
  'audit_postfilter_oracle_macro_f05':score_arrays(a,y,y.astype(float),.5,tc,audit_ids,ret&fast_keep)[0],
  'country_audit':{},'seconds':time.monotonic()-START}
 for c in sorted(set(countries)):
  mask=audit_ids&(countries==c)
  report['country_audit'][c]={'entities':int(mask.sum()),'macro_f05':float(scores[mask].mean()),
   'precision':float(tp[mask].sum()/max(1,pred[mask].sum())),
   'recall':float(tp[mask].sum()/max(1,tc[mask].sum()))}
 # Binomial standard error does not assume observations are Bernoulli: use sample SD.
 se=float(scores[audit_ids].std(ddof=1)/math.sqrt(audit_ids.sum()))
 report['audit_approx_95ci']=[max(0,sc-1.96*se),min(1,sc+1.96*se)]
 with (OUT/'audit_errors.tsv').open('w',encoding='utf-8',newline='') as f:
  w=csv.writer(f,delimiter='\t'); w.writerow(['id','country','name','address','score','truth','predicted'])
  for i in np.flatnonzero(audit_ids&(scores<.999)):
   rr=m['records'][i]; pids=meta.candidate[(a==i)&ret&(prob>=decision)].tolist()
   w.writerow([rr[0],rr[3],rr[1],rr[2],scores[i],','.join(m['truth'][rr[0]]),','.join(pids)])
 with (OUT/'validation_model.pkl').open('wb') as f: pickle.dump(model,f)
 (OUT/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 log(json.dumps(report,indent=2))
 # Refit on fit+tune; retain audit as genuinely unseen in the saved production model.
 final=lgb.LGBMClassifier(**{**params,'n_estimators':max(50,model.best_iteration_)})
 final.fit(X[ng<8],y[ng<8],feature_name=feature_names)
 production_prob=np.zeros(len(y),dtype=np.float32)
 for start in range(0,len(y),100000):
  stop=min(start+100000,len(y)); mask=audit[start:stop]
  if mask.any():
   chunk=np.zeros(stop-start,dtype=np.float32)
   chunk[mask]=probabilities(final,X[start:stop][mask])
   production_prob[start:stop]=chunk
 production_prob[~fast_keep]=0.
 production_score,ps,pp,pt=score_arrays(a,y,production_prob,decision,tc,audit_ids,ret)
 report['production_audit_macro_f05']=production_score
 report['production_country_audit']={c:float(ps[audit_ids&(countries==c)].mean()) for c in sorted(set(countries))}
 report['feature_importance']=dict(sorted(zip(feature_names,map(int,final.feature_importances_)),key=lambda kv:-kv[1]))
 (OUT/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 log(f'Production audit macro F0.5: {production_score:.6f}')
 with (MODELS/'model.pkl').open('wb') as f: pickle.dump({'model':final,'fast_model':fast,'fast_threshold':fast_threshold,'threshold':threshold,'country_thresholds':country_thresholds,'features':feature_names,'version':VERSION},f)
 log('Production model saved (audit entities remain excluded)')
def predict(args):
 model_path=Path(args.model) if args.model else MODELS/'model.pkl'
 with model_path.open('rb') as f: bundle=pickle.load(f)
 schemas=[FEATURES+CONTEXT_FEATURES,FEATURES+CONTEXT_FEATURES+SOURCE_FEATURES+EXTRA_FEATURES]
 if bundle['version']!=VERSION or bundle['features'] not in schemas: raise ValueError('Incompatible model')
 extended=len(bundle['features'])==len(schemas[1])
 model=bundle['model']; threshold=bundle['threshold'] if args.threshold is None else args.threshold
 countries=Counter(r['country'].lower().strip() for r in rows('test',1))
 log(f'Test references: {dict(countries)}; threshold={threshold}')
 signature=hashlib.sha256(model_path.read_bytes()).hexdigest()+f':{threshold}:{args.shard_size}:{VERSION}:'+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 manifest_path=OUT/'inference_manifest.json'
 if manifest_path.exists():
  old=json.loads(manifest_path.read_text())
  if old['signature']!=signature:
   raise ValueError('Inference outputs belong to another model/configuration. Archive them before a new run.')
 manifest_path.write_text(json.dumps({'signature':signature,'countries':dict(countries),'shard_size':args.shard_size}),encoding='utf-8')
 for country in sorted(countries):
  country_threshold=threshold if args.threshold is not None else bundle.get('country_thresholds',{}).get(country,threshold)
  log(f'Building/reusing {country} source frequencies')
  profile=source_profile('test',country) if extended else None
  country_records=[record(r) for r in rows('test',1) if r['country'].lower().strip()==country]
  for start in range(0,len(country_records),args.shard_size):
   recs=country_records[start:start+args.shard_size]
   tag=f'{country}_{start//args.shard_size:03d}'
   done=OUT/f'pred_{tag}.done'
   if args.resume and done.exists(): log(f'Reuse completed {tag}'); continue
   predict_shard(args,model,country_threshold,country,recs,tag,done,len(country_records)>args.shard_size,bundle,profile)
  del country_records,profile; gc.collect()
 assemble(args)
 check()
def predict_shard(args,model,threshold,country,recs,tag,done,restrict,bundle,profile):
  idx=Index(recs)
  if restrict: idx.restrict_global('test',country)
  context=[context_features(r,idx.context_counts) for r in recs]
  log(f'{tag}: built {len(idx.inv):,} keys for {len(recs):,} references')
  # Pair files are streamed; no global candidate DataFrame or source-2/3 cache.
  progress=OUT/f'progress_{tag}.json'
  state=json.loads(progress.read_text()) if args.resume and progress.exists() else {}
  link_path=OUT/f'links_{tag}.tsv'; candidate_path=OUT/f'candidates_{tag}.tsv'
  if state:
   for path,size in [(link_path,state['link_bytes']),(candidate_path,state['candidate_bytes'])]:
    if not path.exists() or path.stat().st_size<size: raise ValueError(f'Incomplete checkpoint file: {path}')
    with path.open('r+b') as f: f.truncate(size)
   log(f'{tag}: resume after {state["records"]:,} source records')
  elif progress.exists(): progress.unlink()
  with link_path.open('a' if state else 'w',encoding='utf-8',newline='') as lf,candidate_path.open('a' if state else 'w',encoding='utf-8',newline='') as cf:
   lw=csv.writer(lf,delimiter='\t'); cw=csv.writer(cf,delimiter='\t')
   pairs=[]; scanned=0; skip=state.get('records',0)
   npairs=state.get('pairs',0); nmatches=state.get('matches',0); screened=state.get('screened',0)
   def flush():
    nonlocal npairs,nmatches,screened
    if not pairs: return
    matrix,rough=fast_matrix(pairs,recs,context,args.threads)
    screened+=int(rough.sum())
    keep=np.zeros(len(pairs),dtype=bool)
    if rough.any(): keep[rough]=probabilities(bundle['fast_model'],matrix[rough])>=bundle['fast_threshold']
    full=[]; kept_pairs=[]
    for (j,r,sc),accept in zip(pairs,keep):
     if accept:
      full.append(features(recs[j],r)+context[j]+sc+(extra_features(recs[j],r) if profile is not None else [])); kept_pairs.append((recs[j][0],r[0]))
    if full:
     probs=probabilities(model,np.asarray(full,dtype=np.float32))
     cw.writerows(kept_pairs)
     for pair,pr in zip(kept_pairs,probs):
      if pr>=threshold: lw.writerow([*pair,float(pr)]); nmatches+=1
    npairs+=len(full); pairs.clear(); lf.flush(); cf.flush()
   for source in [2,3]:
    for r in normalized_rows('test',source,country):
     scanned+=1
     if scanned<=skip: continue
     sc=source_context(r,profile) if profile is not None else []
     for j in idx.query(r):
      pairs.append((j,r,sc))
     if len(pairs)>=50000: flush()
     if scanned%50000==0:
      flush()
      checkpoint={'records':scanned,'pairs':npairs,'matches':nmatches,'screened':screened,
       'link_bytes':lf.tell(),'candidate_bytes':cf.tell()}
      temp=progress.with_suffix('.tmp'); temp.write_text(json.dumps(checkpoint),encoding='utf-8'); commit_output(temp,progress)
      log(f'{tag}: {scanned:,} records, {npairs:,} pairs, {nmatches:,} matches; checkpoint saved')
    flush()
   done.write_text(json.dumps({'records':scanned,'screened':screened,'pairs':npairs,'matches':nmatches,'threshold':threshold}),encoding='utf-8')
  del idx; gc.collect()
def assemble(args):
 # SQLite external grouping bounds memory during conversion to official wide TSVs.
 import sqlite3
 manifest=json.loads((OUT/'inference_manifest.json').read_text())
 expected=[f'{c}_{i:03d}' for c,n in manifest['countries'].items() for i in range(math.ceil(n/manifest['shard_size']))]
 if any(not (OUT/f'pred_{tag}.done').exists() for tag in expected):
  raise ValueError('Inference is incomplete. Resume predict before assembling a submission.')
 db=sqlite3.connect(OUT/'assembly.sqlite'); db.execute('PRAGMA journal_mode=OFF'); db.execute('PRAGMA synchronous=OFF')
 for table,prefix,header in [('matches','links','matched_entity_ids'),('candidates','candidates','candidate_entity_ids')]:
  db.execute(f'DROP TABLE IF EXISTS {table}')
  db.execute(f'CREATE TABLE {table}(anchor TEXT, candidate TEXT, PRIMARY KEY(anchor,candidate)) WITHOUT ROWID')
  for path in [OUT/f'{prefix}_{tag}.tsv' for tag in sorted(expected)]:
   with path.open(encoding='utf-8',newline='') as f:
    batch=[]
    for row in csv.reader(f,delimiter='\t'):
     batch.append(row[:2])
     if len(batch)>=50000:
      db.executemany(f'INSERT OR IGNORE INTO {table} VALUES (?,?)',batch); db.commit(); batch=[]
    if batch: db.executemany(f'INSERT OR IGNORE INTO {table} VALUES (?,?)',batch); db.commit()
  filename='matching_results.tsv' if table=='matches' else 'candidate_pairs.tsv'
  with (OUT/(filename+'.tmp')).open('w',encoding='utf-8',newline='') as f:
   w=csv.writer(f,delimiter='\t'); w.writerow(['source1_entity_id',header])
   for raw in rows('test',1):
    mids=[r[0] for r in db.execute(f'SELECT candidate FROM {table} WHERE anchor=?',(raw['entity_id'],))]
    w.writerow([raw['entity_id'],','.join(mids)])
  commit_output(OUT/(filename+'.tmp'),RESULTS/filename)
  log(f'Wrote {filename}')
 db.close()
def diagnose(args):
 import pandas as pd
 m=json.loads((OUT/'training_manifest.json').read_text(encoding='utf-8'))
 met=json.loads((OUT/'metrics.json').read_text()); t=met['threshold']
 df=pd.read_csv(OUT/'pairs.tsv',sep='\t'); p=np.load(OUT/'validation_probabilities.npy')
 found=defaultdict(set); predicted=defaultdict(set)
 for a,k,ret,pr in zip(df.anchor,df.candidate,df.retrieved,p):
  if ret: found[a].add(k)
  if ret and pr>=t: predicted[a].add(k)
 examples=[]; wanted=set()
 for i,r in enumerate(m['records']):
  if m['groups'][i] not in [6,7]: continue
  true=set(m['truth'][r[0]]); missing=true-found[i]; fn=true-predicted[i]; fp=predicted[i]-true
  if missing or fn or fp:
   examples.append({'reference':r,'retrieval_misses':sorted(missing),'false_negatives':sorted(fn),'false_positives':sorted(fp)})
   wanted.update(missing|fn|fp)
  if len(examples)>=100: break
 records={}
 for source in [2,3]:
  for raw in rows('train',source):
   if raw['entity_id'] in wanted: records[raw['entity_id']]=raw
 (OUT/'development_errors.json').write_text(json.dumps({'examples':examples,'records':records},indent=2),encoding='utf-8')
 log(f'Saved {len(examples)} development examples; {len(records)} related records')
def assess(args):
 data=json.loads((OUT/'retrieval_misses.json').read_text(encoding='utf-8'))
 recovered=0; total=0; gated=0
 for country in sorted({e['reference'][3] for e in data['examples']}):
  examples=[e for e in data['examples'] if e['reference'][3]==country]
  idx=Index([e['reference'] for e in examples]); idx.restrict_global('train',country)
  for j,e in enumerate(examples):
   for k in e['missing']:
    r=record(data['records'][k]); hit=j in idx.query(r); total+=1
    recovered+=int(hit and gate(e['reference'],r)); gated+=int(hit and not gate(e['reference'],r))
  log(f'{country}: cumulative recovered={recovered}/{total}; rejected by broad filter={gated}')
 (OUT/'retrieval_assessment.json').write_text(json.dumps({'recovered':recovered,'total':total,'gate_rejects':gated}),encoding='utf-8')
def benchmark(args):
 import psutil
 recs=[]
 for raw in rows('test',1):
  if raw['country']=='India': recs.append(record(raw))
  if len(recs)>=args.shard_size: break
 idx=Index(recs); log(f'Index built: RSS={psutil.Process().memory_info().rss/1e9:.2f} GB')
 idx.restrict_global('test','india'); log(f'Index filtered: {len(idx.inv):,} keys; RSS={psutil.Process().memory_info().rss/1e9:.2f} GB')
 n=0; pairs=0; t=time.monotonic()
 for r in normalized_rows('test',2,'india'):
  for j in idx.query(r):
   if gate(recs[j],r):
    fast_features(recs[j],r,[0.,0.,0.,0.,0.]); pairs+=1
  n+=1
  if n>=50000: break
 result={'references':len(recs),'records':n,'pairs':pairs,'seconds':time.monotonic()-t,'rss_gb':psutil.Process().memory_info().rss/1e9}
 (OUT/'benchmark.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); log(json.dumps(result))
def pilot(args):
 import cProfile,pstats,itertools
 with (MODELS/'model.pkl').open('rb') as f: bundle=pickle.load(f)
 profile=source_profile('test','france')
 recs=[record(r) for r in rows('test',1) if r['country']=='France']
 original=normalized_rows
 globals()['normalized_rows']=lambda split,source,country: itertools.islice(original(split,source,country),5000)
 pr=cProfile.Profile(); pr.enable()
 predict_shard(args,bundle['model'],bundle['threshold'],'france',recs,'pilot_france',OUT/'pilot_france.done',False,bundle,profile)
 pr.disable(); pr.dump_stats(str(OUT/'pilot.prof')); pstats.Stats(pr).sort_stats('cumtime').print_stats(20)
def selftest():
 assert clean('Café & Fils')=='cafe and fils'
 x=record(dict(entity_id='S1-a',business_name='Acme Pvt Ltd',business_address='12 Main Road',country='India'))
 y=record(dict(entity_id='S2-a',business_name='ACME PRIVATE LIMITED',business_address='12 Main Rd',country='India'))
 assert x[1:]==y[1:]; assert len(features(x,y))==len(FEATURES)
 assert len(set(FEATURES+CONTEXT_FEATURES))==len(FEATURES+CONTEXT_FEATURES)
 assert len(extra_features(x,y))==len(EXTRA_FEATURES)
 assert np.allclose(np.array(features(x,y)+[1.,2.,3.,4.,5.])[FAST_COLS],fast_features(x,y,[1.,2.,3.,4.,5.]))
 fm,gm=fast_matrix([(0,y,[4.,5.])],[x],[[1.,2.,3.]],2)
 assert np.allclose(fm[0],fast_features(x,y,[1.,2.,3.,4.,5.])) and gm[0]==gate(x,y)
 variants=[y,('S3-z','','','',''),('S2-z','different','12 main street','india','different'),
  ('S2-z','acme','', 'india','acme'),('S2-z','','12 main road','india','')]
 fm,gm=fast_matrix([(0,r,[4.,5.]) for r in variants],[x],[[1.,2.,3.]],2)
 for i,r in enumerate(variants):
  assert np.array_equal(fm[i],np.asarray(fast_features(x,r,[1.,2.,3.,4.,5.]),dtype=np.float32)) and gm[i]==gate(x,r)
 assert Index([x]).query(y)=={0}
 a=np.array([0,0,1]); labels=np.array([1,0,0]); prob=np.array([.9,.2,.1]); tc=np.array([2,0,1])
 sc,s,_,_=score_arrays(a,labels,prob,.5,tc,np.ones(3,dtype=bool))
 assert np.allclose(s,[1.25/1.5,1,0])
 assert features(('', '', '', '', ''),('', '', '', '', ''))[28]==0
 log('Self-tests passed: normalization, feature schema, retrieval, singleton/missed-truth metric')
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--step',choices=['prepare','fit','predict','assemble','check','diagnose','assess','profile','profile_test','benchmark','pilot','selftest','all'],default='all')
 p.add_argument('--sample-rate',type=float,default=.008,help='Fraction of train reference entities; all source 2/3 records are scanned')
 p.add_argument('--threads',type=int,default=8); p.add_argument('--threshold',type=float)
 p.add_argument('--model',help='Optional saved model bundle path for inference')
 p.add_argument('--shard-size',type=int,default=1000000,help='Maximum reference entities in each production index')
 p.add_argument('--resume',action='store_true',help='Reuse completed inference countries only with the same model/threshold')
 args=p.parse_args()
 if args.step=='selftest': selftest(); return
 if args.step in ['prepare','all']: prepare(args)
 if args.step in ['fit','all']: fit(args)
 if args.step in ['predict','all']: predict(args)
 if args.step=='assemble': assemble(args)
 if args.step=='check': check()
 if args.step=='diagnose': diagnose(args)
 if args.step=='assess': assess(args)
 if args.step=='benchmark': benchmark(args)
 if args.step=='pilot': pilot(args)
 if args.step=='profile':
  for country in ['india','us']:
   log(f'Profiling train {country}')
   profile=source_profile('train',country); del profile; gc.collect()
   log(f'Profiled train {country}')
 if args.step=='profile_test':
  for country in ['france','india','us']:
   log(f'Profiling test {country}')
   profile=source_profile('test',country); del profile; gc.collect()
   log(f'Profiled test {country}')
if __name__=='__main__': main()
