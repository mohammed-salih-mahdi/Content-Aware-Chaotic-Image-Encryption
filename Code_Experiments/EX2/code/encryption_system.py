"""Revised Round-2 research implementation.

The chaotic core is the established 4-D hyperchaotic Lorenz system:
 x'=a(y-x)+w, y'=cx-y-xz, z'=xy-bz, w'=-yz+rw.

Protocol: a random 256-bit root key K is the only secret. SHA-256(P) is a
non-secret content-binding value. HKDF-SHA256(K, nonce, SHA-256(P)) derives the
chaotic initial state and a deterministic Cuckoo-Search seed. The selected
parameters and initial state are protected inside an AES-256-GCM metadata
record. HMAC-SHA256 authenticates the complete package. Decryption therefore
never needs plaintext before decryption.
"""
import base64,hashlib,hmac,json,os,time
from pathlib import Path
import cv2, numpy as np, matplotlib.pyplot as plt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from skimage.metrics import structural_similarity as ssim
from cuckoo_search import CuckooSearch
from nist_randomness_tests import (nist_01_monobit,nist_02_block_frequency,nist_03_runs,nist_04_longest_run_of_ones,nist_05_binary_matrix_rank,nist_06_spectral_dft,nist_07_nonoverlapping_template,nist_08_overlapping_template,nist_09_universal,nist_10_linear_complexity,nist_11_serial,nist_12_approximate_entropy,nist_13_cumulative_sums)

VERSION='2.0'; KEY_LEN=32; NONCE_LEN=12

def b64e(x): return base64.b64encode(x).decode()
def b64d(x): return base64.b64decode(x)

class FullResearchEncryptionSystem:
    def __init__(self,image_path,output_folder='Final_Research_Data',secret_key=None,cs_config=None):
        self.image_path=image_path; self.output_folder=output_folder; Path(output_folder).mkdir(exist_ok=True)
        img=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
        if img is None: raise FileNotFoundError(image_path)
        if img.ndim==2: img=img[:,:,None]
        self.img=img.astype(np.uint8); self.rows,self.cols,self.channels=self.img.shape
        self.secret_key=os.urandom(KEY_LEN) if secret_key is None else bytes(secret_key)
        if len(self.secret_key)!=KEY_LEN: raise ValueError('secret_key must be 32 bytes')
        self.cs_config={'pop_size':6,'pa':0.25,'max_iter':1,'levy_beta':1.5,'quick_steps':500,'dt':0.001}
        if cs_config:self.cs_config.update(cs_config)
        self.dt=float(self.cs_config['dt']); self.params=None; self.initials=None; self.cs_time=0

    @staticmethod
    def sha256(data): return hashlib.sha256(np.ascontiguousarray(data).tobytes()).digest()
    def hkdf(self,salt,info,n): return HKDF(algorithm=hashes.SHA256(),length=n,salt=salt,info=info).derive(self.secret_key)
    def derive_initials(self,digest,nonce):
        raw=self.hkdf(nonce,b'FB-CHAOS/v2/initials/'+digest,32); u=[int.from_bytes(raw[i*8:(i+1)*8],'big')/2**64 for i in range(4)]
        # Stay close to the standard reference basin while retaining full
        # content/key dependence; this avoids changing the dynamical regime.
        base=np.array([-.1,.5,-.6,.2]); scale=np.array([.02,.02,.02,.02])
        return base+(np.array(u)-.5)*scale
    def cs_seed(self,digest,nonce): return int.from_bytes(self.hkdf(nonce,b'FB-CHAOS/v2/cs-seed/'+digest,8),'big')%(2**32)
    def subkeys(self,nonce):
        x=self.hkdf(nonce,b'FB-CHAOS/v2/subkeys',64); return x[:32],x[32:]

    # ---- continuous dynamics ----
    @staticmethod
    def f(s,p):
        x,y,z,w=s; a,b,c,r=p; return np.array([a*(y-x)+w,c*x-y-x*z,x*y-b*z,-y*z+r*w],float)
    @staticmethod
    def J(s,p):
        x,y,z,w=s; a,b,c,r=p; return np.array([[-a,a,0,1],[c-z,-1,-x,0],[y,x,-b,0],[0,-z,-y,r]],float)
    @staticmethod
    def divergence(p): a,b,c,r=p; return -a-1-b+r
    def rk4(self,s,p,h=None):
        h=self.dt if h is None else h; k1=self.f(s,p);k2=self.f(s+.5*h*k1,p);k3=self.f(s+.5*h*k2,p);k4=self.f(s+h*k3,p);return s+h*(k1+2*k2+2*k3+k4)/6
    def lyapunov(self,initials,p,transient=5000,steps=50000,dt=None):
        h=self.dt if dt is None else dt;s=np.array(initials,float)
        for _ in range(transient): s=self.rk4(s,p,h)
        V=np.eye(4);acc=np.zeros(4)
        for _ in range(steps):
            def rhs(x,Y):return self.f(x,p),self.J(x,p)@Y
            k1s,k1v=rhs(s,V);k2s,k2v=rhs(s+.5*h*k1s,V+.5*h*k1v);k3s,k3v=rhs(s+.5*h*k2s,V+.5*h*k2v);k4s,k4v=rhs(s+h*k3s,V+h*k3v)
            s=s+h*(k1s+2*k2s+2*k3s+k4s)/6;V=V+h*(k1v+2*k2v+2*k3v+k4v)/6
            if not np.all(np.isfinite(s)) or np.linalg.norm(s)>1e5: raise RuntimeError('Divergent trajectory')
            Q,R=np.linalg.qr(V);d=np.abs(np.diag(R))
            if np.any(d<=0):raise RuntimeError('Degenerate QR')
            acc+=np.log(d);V=Q
        return acc/(steps*h)
    def dynamics_check(self,le,p):
        le=np.asarray(le); total=float(le.sum());div=float(self.divergence(p));diff=total-div
        return {'lyapunov':le.tolist(),'sum_lyapunov':total,'divergence':div,'difference':diff,'positive_exponents':int((le>1e-4).sum()),'dissipative':div<0,'hyperchaotic':int((le>1e-4).sum())>=2 and div<0 and abs(diff)<0.05}

    def optimize_parameters(self,image,nonce):
        digest=self.sha256(image); initials=self.derive_initials(digest,nonce); seed=self.cs_seed(digest,nonce)
        t=time.perf_counter(); cs=CuckooSearch(initials,seed=seed,**self.cs_config); best,fit=cs.optimize(False)
        candidates=cs.candidate_pool
        # Authoritative validation: rank the CS candidates, screen the top three
        # with an independent 10k-step spectrum, then report a 50k-step spectrum.
        # This avoids spending the full validation cost on every inferior nest.
        for p in candidates[:3]:
            try:
                le10=self.lyapunov(initials,p,2000,10000,self.dt); chk10=self.dynamics_check(le10,p)
                if not chk10['hyperchaotic']:continue
                le50=self.lyapunov(initials,p,5000,50000,self.dt); chk50=self.dynamics_check(le50,p)
                if chk50['hyperchaotic']:
                    self.params=np.array(p);self.initials=initials;self.cs_time=time.perf_counter()-t;return digest,seed,le50,chk50
            except RuntimeError:continue
        raise RuntimeError('Cuckoo Search did not produce a candidate that passes independent 10k/50k LE validation')

    # ---- chaotic byte streams ----
    def streams(self,n,initials,p):
        s=np.array(initials,float);kp=np.empty(n);kd=np.empty(n,dtype=np.uint8)
        for i in range(n):
            s=self.rk4(s,p);kp[i]=s[0]
            q=np.floor(np.abs(s)*np.array([1e6,1e5,1e4,1e3])).astype(np.uint64)&255
            kd[i]=np.bitwise_xor.reduce(q.astype(np.uint8))
        return kp,kd

    # ---- authenticated protocol ----
    def enc_meta(self,meta,nonce):
        k,_=self.subkeys(nonce); aad=b'FB-CHAOS-META-v2'+nonce
        raw=json.dumps(meta,sort_keys=True,separators=(',',':')).encode();return AESGCM(k).encrypt(nonce,raw,aad)
    def dec_meta(self,blob,nonce):
        k,_=self.subkeys(nonce);return json.loads(AESGCM(k).decrypt(nonce,blob,b'FB-CHAOS-META-v2'+nonce).decode())
    def encrypt_core(self,image,nonce,params):
        """Encrypt using supplied validated parameters without re-running CS.
        Used only for controlled differential/key-sensitivity experiments."""
        digest=self.sha256(image); ini=self.derive_initials(digest,nonce); kp,kd=self.streams(image.size,ini,np.asarray(params,float))
        flat=np.asarray(image,dtype=np.uint8).reshape(-1);idx=np.argsort(kp,kind='mergesort')
        return np.bitwise_xor(flat[idx],kd).reshape(image.shape)

    def encrypt_image(self,image=None,nonce=None):
        image=self.img if image is None else np.asarray(image,dtype=np.uint8); nonce=os.urandom(NONCE_LEN) if nonce is None else bytes(nonce)
        digest,seed,le,dyn=self.optimize_parameters(image,nonce);kp,kd=self.streams(image.size,self.initials,self.params)
        flat=image.reshape(-1);idx=np.argsort(kp,kind='mergesort');cipher=np.bitwise_xor(flat[idx],kd).reshape(image.shape)
        meta={'version':VERSION,'shape':list(image.shape),'sha256':digest.hex(),'nonce':nonce.hex(),'cs_seed':int(seed),'parameters':self.params.tolist(),'initials':self.initials.tolist(),'dt':self.dt,'lyapunov':le.tolist(),'divergence':float(self.divergence(self.params)),'cs_config':self.cs_config}
        blob=self.enc_meta(meta,nonce);_,mk=self.subkeys(nonce);tag=hmac.new(mk,nonce+blob+cipher.tobytes(),hashlib.sha256).digest()
        return cipher,{'version':VERSION,'nonce':b64e(nonce),'metadata':b64e(blob),'hmac':b64e(tag)},meta
    def decrypt_image(self,cipher,package,secret_key=None,verify=True):
        if secret_key is not None:self.secret_key=bytes(secret_key)
        nonce=b64d(package['nonce']);blob=b64d(package['metadata']);given=b64d(package['hmac']);_,mk=self.subkeys(nonce)
        expected=hmac.new(mk,nonce+blob+np.asarray(cipher,dtype=np.uint8).tobytes(),hashlib.sha256).digest()
        if verify and not hmac.compare_digest(given,expected):raise ValueError('Authentication failed')
        meta=self.dec_meta(blob,nonce);p=np.array(meta['parameters']);ini=np.array(meta['initials']);self.params=p;self.initials=ini;self.dt=float(meta['dt'])
        kp,kd=self.streams(cipher.size,ini,p);diff=np.bitwise_xor(np.asarray(cipher,dtype=np.uint8).reshape(-1),kd);idx=np.argsort(kp,kind='mergesort');plain=np.empty(cipher.size,dtype=np.uint8);plain[idx]=diff;plain=plain.reshape(tuple(meta['shape']))
        if verify and not hmac.compare_digest(self.sha256(plain).hex(),meta['sha256']):raise ValueError('Plaintext integrity check failed')
        return plain,meta
    @staticmethod
    def save_package(package,path):
        with open(path,'w') as f:json.dump(package,f,indent=2)
    @staticmethod
    def load_package(path):
        with open(path) as f:return json.load(f)

    # ---- empirical security/statistics ----
    @staticmethod
    def npcr_uaci(a,b):
        a=np.asarray(a,dtype=np.int32);b=np.asarray(b,dtype=np.int32);return float(np.mean(a!=b)*100),float(np.mean(np.abs(a-b))/255*100)
    @staticmethod
    def entropy(image):
        out=[]
        for c in range(image.shape[2]):
            q=np.bincount(image[:,:,c].ravel(),minlength=256).astype(float);p=q[q>0]/image[:,:,c].size;out.append(float(-np.sum(p*np.log2(p))))
        return out
    @staticmethod
    def corr(image):
        out={}
        for c,n in enumerate(['B','G','R'][:image.shape[2]]):
            p=image[:,:,c].astype(float);h,w=p.shape;out[n]={}
            for name,(dy,dx) in {'H':(0,1),'V':(1,0),'D':(1,1)}.items():out[n][name]=float(np.corrcoef(p[:h-dy or h,:w-dx or w].ravel(),p[dy:,dx:].ravel())[0,1])
        return out
    @staticmethod
    def local_entropy(image,window=32):
        v=[]
        for y in range(0,image.shape[0]-window+1,window):
            for x in range(0,image.shape[1]-window+1,window):
                q=np.bincount(image[y:y+window,x:x+window].ravel(),minlength=256).astype(float);p=q[q>0]/q.sum();v.append(-np.sum(p*np.log2(p)))
        return (float(np.mean(v)),float(np.std(v))) if v else (0.,0.)
    @staticmethod
    def metrics(a,b):
        mse=np.mean((a.astype(float)-b.astype(float))**2);psnr=float(20*np.log10(255/np.sqrt(mse))) if mse else float('inf');sv=float(np.mean([ssim(a[:,:,c],b[:,:,c],data_range=255) for c in range(a.shape[2])]))
        return psnr,sv
    def key_sensitivity(self,cipher,package):
        meta=self.dec_meta(b64d(package['metadata']),b64d(package['nonce']));p=np.array(meta['parameters']);ini=np.array(meta['initials']);correct,_=self.decrypt_image(cipher,package,verify=True)
        pert=ini.copy();pert[0]=pert[0]+1e-12*max(1.0,abs(pert[0]));kp,kd=self.streams(cipher.size,pert,p);d=np.bitwise_xor(cipher.reshape(-1),kd);idx=np.argsort(kp,kind='mergesort');wrong=np.empty(cipher.size,dtype=np.uint8);wrong[idx]=d;wrong=wrong.reshape(cipher.shape);ps,ss=self.metrics(correct,wrong);npcr,uaci=self.npcr_uaci(correct,wrong)
        bad=bytearray(self.secret_key);bad[0]^=1
        try:self.decrypt_image(cipher,package,bytes(bad),True);rejected=False
        except ValueError:rejected=True
        return {'one_ulp_initial_state':{'PSNR_dB':ps,'SSIM':ss,'NPCR_percent':npcr,'UACI_percent':uaci},'one_bit_root_key_rejected':rejected}
    def differential(self,image,cipher,package):
        # One-pixel plaintext change, same root key and same nonce/selected
        # parameter vector. The content-bound initial state changes through
        # SHA-256, so this isolates the plaintext-binding/diffusion effect
        # without rerunning the metaheuristic.
        m=image.copy();m.reshape(-1)[0]^=1
        nonce=b64d(package['nonce']);p=np.array(self.dec_meta(b64d(package['metadata']),nonce)['parameters'])
        c2=self.encrypt_core(m,nonce,p); n,u=self.npcr_uaci(cipher,c2)
        return {'NPCR_percent':n,'UACI_percent':u,'protocol_note':'same root key/nonce/selected parameters; modified plaintext gets a different content-bound initial state'}

    def nist(self,byte_stream):
        # NIST SP 800-22 tests 1-13 are reported here. Tests 14-15 require
        # a sufficient number of random-walk cycles and are intentionally
        # not fabricated when the generated stream does not meet that condition.
        bits=np.unpackbits(np.frombuffer(byte_stream,dtype=np.uint8)); a=.01; r={}
        r['01. Frequency (Monobit)']=nist_01_monobit(bits)
        r['02. Block Frequency']=nist_02_block_frequency(bits)
        r['03. Runs']=nist_03_runs(bits)
        r['04. Longest Run of Ones in a Block']=nist_04_longest_run_of_ones(bits)
        r['05. Binary Matrix Rank']=nist_05_binary_matrix_rank(bits)
        r['06. Discrete Fourier Transform (Spectral)']=nist_06_spectral_dft(bits)
        r['07. Non-overlapping Template Matching']=nist_07_nonoverlapping_template(bits)
        r['08. Overlapping Template Matching']=nist_08_overlapping_template(bits)
        r['09. Maurer Universal Statistical']=nist_09_universal(bits)
        r['10. Linear Complexity']=nist_10_linear_complexity(bits)
        p1,p2=nist_11_serial(bits);r['11a. Serial (delta psi^2)']=p1;r['11b. Serial (delta^2 psi^2)']=p2
        r['12. Approximate Entropy']=nist_12_approximate_entropy(bits)
        r['13a. Cumulative Sums (forward)']=nist_13_cumulative_sums(bits,'forward')
        r['13b. Cumulative Sums (backward)']=nist_13_cumulative_sums(bits,'backward')
        return r,{k:('PASS' if v is not None and v>=a else 'FAIL' if v is not None else 'N/A') for k,v in r.items()}

    def run_full_analysis(self):
        tag=Path(self.image_path).stem;out=Path(self.output_folder)
        t=time.perf_counter();cipher,package,meta=self.encrypt_image(self.img);enc=time.perf_counter()-t
        t=time.perf_counter();plain,_=self.decrypt_image(cipher,package);dec=time.perf_counter()-t
        if not np.array_equal(self.img,plain):raise AssertionError('round trip failed')
        cv2.imwrite(str(out/f'{tag}_Encrypted.png'),cipher);cv2.imwrite(str(out/f'{tag}_Decrypted.png'),plain);self.save_package(package,out/f'{tag}_package.json')
        diff=self.differential(self.img,cipher,package);ks=self.key_sensitivity(cipher,package);ent=self.entropy(cipher);corr=self.corr(cipher);loc=self.local_entropy(cipher)
        _,kd=self.streams(min(cipher.size,125000),self.initials,self.params);nr,nv=self.nist(kd.tobytes())
        report={'version':VERSION,'shape':list(self.img.shape),'encryption_time_sec':enc,'decryption_time_sec':dec,'parameters':self.params.tolist(),'lyapunov':meta['lyapunov'],'divergence':meta['divergence'],'lyapunov_sum_difference':float(sum(meta['lyapunov'])-meta['divergence']),'differential_test':diff,'key_sensitivity':ks,'entropy_per_channel':ent,'mean_entropy':float(np.mean(ent)),'correlation':corr,'local_entropy_mean_std':loc,'nist_results':nr,'nist_verdict':nv,'secret_key_bits':256,'security_claim':'Empirical tests only; no formal KPA/CPA proof claimed.'}
        with open(out/f'Research_Report_{tag}.json','w') as f:json.dump(report,f,indent=2,default=float)
        return str(out/f'Research_Report_{tag}.json')
