# 模型装配、训练与实际板端调用摘录

以下摘录自审查时本机源码，左侧为原文件行号。同一文件内分段有意省略未引用行。文件 SHA-256 是原完整文件的指纹，不是摘录的指纹。H0 归档绑定了 `RunQiandaoHistoricalClockAB.py` 的相同 SHA-256；其余文件摘要仅标识所审阅的本机源码，不能当作各次运行的全部设备源码。此为核对资料，不是可运行发行包。

## 三权重装配、静态 ONNX 签名：PrepareQiandaoRecordedSmallModelAB.py

本机源码：`standalone/PrepareQiandaoRecordedSmallModelAB.py`；原行号：21–55、74–91；原文件 SHA-256：`19b682c17e4cad7f699bdee9a13ea362b4dfd9771ead3134467fcdb80a9ce4a5`。

```text
    21
    22	PROTECTED='3fbf8d28f67f7e3f6d818507fede12715c43e98909d681cc6a6f533cf6c7a5c6'
    23	INPUTS=('features','positive','crossfit','state0','state1','state2','state3')
    24	OUTPUTS=('gain','next_state0','next_state1','next_state2','next_state3')
    25	REAL_CAL=dict(sound_speed_mps=1480.,sample_rate_hz=37650.,row_spacing_m=.15)
    26
    27
    28	def load_model(path,device):
    29	    keys=json.loads((path/'weights_sha.json').read_text())
    30	    if keys['protection.pt']!=PROTECTED:
    31	        raise ValueError('Only the 7.52M model that rendered the original full survey.')
    32	    def w(name):
    33	        f=path/name
    34	        if sha(f)!=keys[name]:
    35	            raise ValueError('The staged checkpoint does not match its originally verified weights.')
    36	        return torch.load(f,map_location='cpu',weights_only=True)['model']
    37	    a=LateSpectralJointBeam(128);a.load_parent_and_readout(w('wavelet.pt'),w('spectral.pt'))
    38	    b=WeakEchoVerifier();b.load_state_dict(w('protection.pt'),strict=True)
    39	    return a.eval().requires_grad_(False).to(device),b.eval().requires_grad_(False).to(device)
    40
    41
    42	class FP32LearnedStep(nn.Module):
    43	    """Complex cue construction stays on CUDA outside the real-valued FP32 engine."""
    44	    def __init__(self,net,head,native):
    45	        super().__init__()
    46	        self.net,self.head,self.native=net,head,native
    47
    48	    def forward(self,f,mask,crossfit,a,b,c,d):
    49	        z,states=self.net.step(f,mask,(a,b,c,d),3)
    50	        n=self.native
    51	        data=f[:,:,:,:n].permute(0,2,3,1)
    52	        old=z[:,0,:,:n]
    53	        design=torch.cat((data[...,:4].tanh(),old.tanh()[...,None],data[...,4:]),-1)
    54	        value=self.head(design,crossfit,old)
    55	        return (torch.where(mask[:,0,:,:n].bool(),value,0.).reshape(16,n),*states)
    74	def export(onnx_at,weights):
    75	    import onnx
    76	    a,b=load_model(weights,'cpu')
    77	    exports=[]
    78	    for n in (12189,12721):
    79	        padded=n+(-n)%8
    80	        args=(torch.zeros(16,28,1,padded),torch.ones(16,1,1,padded),
    81	              torch.zeros(16,1,n,6),*a.initial_states(16,padded))
    82	        fname=onnx_at/f'full16_R{n}_learned_fp32.onnx'
    83	        torch.onnx.export(FP32LearnedStep(a,b,n),args,str(fname),input_names=list(INPUTS),output_names=list(OUTPUTS),
    84	                          opset_version=17,dynamo=False,do_constant_folding=True)
    85	        e=onnx.load(str(fname))
    86	        onnx.checker.check_model(e)
    87	        if [x.name for x in e.graph.input]!=list(INPUTS) or [x.name for x in e.graph.output]!=list(OUTPUTS):
    88	            raise ValueError('Exact declared real FP32 runtime schema required.')
    89	        exports.append(dict(path=fname.name,sha256=sha(fname),R=n))
    90	    return exports
    91
```

## 因果递归与距离 Haar 基础结构：CausalSonarWavelet.py

本机源码：`standalone/CausalSonarWavelet.py`；原行号：17–73；原文件 SHA-256：`6f4d1eaa8ab7ddfca822dca76c047d641d40cad86c183245aa1720125dd2e1b5`。

```text
    17	FEATURE_PROTOCOL = 'past4_current_masked_logcontrast_actual_range_phase_v1'
    18
    19
    20	def range_haar(x):
    21	    a, b = x[..., 0::2].float(), x[..., 1::2].float()
    22	    return torch.cat(((a + b) / math.sqrt(2), (a - b) / math.sqrt(2)), dim=1)
    23
    24
    25	def inverse_range_haar(x):
    26	    low, high = x.float().chunk(2, dim=1)
    27	    return torch.stack(((low + high) / math.sqrt(2), (low - high) / math.sqrt(2)), dim=-1).flatten(-2)
    28
    29
    30	def conv(cin, cout, kernel=1, groups=1, dilation=1):
    31	    # Height1 keeps TensorRT's convolution implementation explicit and avoids
    32	    # unnecessary temporal rows. All filtering is learned feature processing,
    33	    # not a received-signal FIR pre/post denoiser.
    34	    return nn.Conv2d(cin, cout, (1, kernel), padding=(0, dilation * (kernel // 2)),
    35	                     groups=groups, dilation=(1, dilation))
    36
    37
    38	class SpatialBlock(nn.Module):
    39	    def __init__(self, channels, dilation=1):
    40	        super().__init__()
    41	        self.depthwise = conv(channels, channels, 7, groups=channels, dilation=dilation)
    42	        self.mix = conv(channels, channels)
    43
    44	    def forward(self, value):
    45	        return value + self.mix(F.relu(self.depthwise(value))) * .2
    46
    47
    48	class CausalFusion(nn.Module):
    49	    def __init__(self, channels):
    50	        super().__init__()
    51	        self.gates = conv(channels * 2, channels * 2)
    52	        self.candidate = conv(channels * 2, channels, 3)
    53
    54	    def forward(self, current, previous, valid):
    55	        reset, retain = torch.sigmoid(self.gates(torch.cat((current, previous), dim=1))).chunk(2, dim=1)
    56	        proposal = torch.tanh(self.candidate(torch.cat((current, previous * reset), dim=1)))
    57	        return ((1 - retain) * proposal + retain * previous) * valid
    58
    59
    60	class CausalSonarWavelet(nn.Module):
    61	    family = FAMILY
    62	    feature_protocol = FEATURE_PROTOCOL
    63
    64	    def __init__(self, width=24):
    65	        super().__init__()
    66	        if type(width) is not int or width < 4 or width % 4:
    67	            raise ValueError('A positive factor-four base width is required.')
    68	        self.width = width
    69	        self.channels = (width, width * 2, width * 3, width * 4)
    70	        self.domain_affine = nn.Embedding(4, 6)
    71	        nn.init.zeros_(self.domain_affine.weight)
    72	        self.head = conv(3, width, 7)
    73	        self.spatial = nn.ModuleList(nn.Sequential(SpatialBlock(c), SpatialBlock(c, 2)) for c in self.channels)
```

## 同舷束输入、四尺度和增益尾部：BaselineAugmentedJointBeam.py

本机源码：`standalone/BaselineAugmentedJointBeam.py`；原行号：48–76；原文件 SHA-256：`3a21e37a34a56b7493041825952c9bb36ed5b03e80562295f33888d2ef76f00b`。

```text
    48	        status=self.load_state_dict(parameters,strict=False)
    49	        if status.missing_keys!=['baseline_head.weight'] or status.unexpected_keys:
    50	            raise RuntimeError('Only the newly initialized cue may be missing.')
    51	        nn.init.zeros_(self.baseline_head.weight)
    52
    53	    def _log_gain_readout(self,latent,domain):
    54	        return self.tail(latent)
    55
    56	    def step(self,features,positive,states,domain=3):
    57	        if (domain not in range(4) or features.ndim!=4 or features.shape[1:3]!=(4,1)
    58	                or features.shape[0]<BEAMS or features.shape[0]%BEAMS
    59	                or positive.shape!=features[:,:1].shape or features.shape[-1]%8 or len(states)!=4):
    60	            raise ValueError('One full native ping, four features and four explicit states required.')
    61	        groups,width=features.shape[0]//BEAMS,features.shape[-1]
    62	        if any(tuple(s.shape)!=(groups,c,1,width//2**i) for i,(s,c) in enumerate(zip(states,self.channels))):
    63	            raise ValueError('Correct independent latent state per side required.')
    64	        native=features[:,:3].float().reshape(groups,BEAMS*3,1,width)
    65	        valid_beam=positive.float().reshape(groups,BEAMS,1,width)
    66	        # Preserve exactly the existing learned domain affine on the original
    67	        # path. The fourth channel is newly learnable, neutral at load time.
    68	        gain,bias=self.domain_affine.weight[domain].float().chunk(2)
    69	        conditioned=native*(1+gain[None,:,None,None])+bias[None,:,None,None]
    70	        auxiliary=torch.where(positive.bool(),features[:,3:4].float(),0.).reshape(groups,BEAMS,1,width)
    71	        x=self.head(torch.cat((conditioned,valid_beam),dim=1))+self.baseline_head(auxiliary)
    72	        support=valid_beam.amax(dim=1,keepdim=True)
    73	        encoded,updated=[],[]
    74	        for i in range(4):
    75	            x=self.spatial[i](x)
    76	            state=self.temporal[i](x,states[i],support)
```

## 晚期频谱读出不修改递归状态：LateSpectralJointBeam.py

本机源码：`standalone/LateSpectralJointBeam.py`；原行号：11–36；原文件 SHA-256：`b5fca0f1527c5d0bbed386fe4da641fd31ed7838e9b3643ba6a7fc6db29eeb98`。

```text
    11
    12	class LateSpectralJointBeam(BaselineAugmentedJointBeam):
    13	    family=FAMILY
    14
    15	    def __init__(self,width=128,*,use_spectrum=True):
    16	        super().__init__(width)
    17	        self.use_spectrum=bool(use_spectrum)
    18	        self.late_head=Readout()
    19
    20	    def load_parent_and_readout(self,parent,readout):
    21	        r=self.load_state_dict(parent,strict=False)
    22	        required={k for k in self.state_dict() if k.startswith('late_head.')}
    23	        if set(r.missing_keys)!=required or r.unexpected_keys:raise ValueError('Exact original four-feature causal state required.')
    24	        self.late_head.load_state_dict(readout,strict=True)
    25
    26	    def step(self,features,positive,states,domain=3):
    27	        if features.ndim!=4 or features.shape[1:3]!=(FEATURES,1):
    28	            raise ValueError('Supply explicit same-ping native spectral planes.')
    29	        old=features[:,:4]
    30	        base,updated=super().step(old,positive,states,domain)
    31	        if domain!=3:return base,updated
    32	        bands=features[:,4:].permute(0,2,3,1)
    33	        if not self.use_spectrum:bands=torch.zeros_like(bands)
    34	        design=torch.cat((old.permute(0,2,3,1).tanh(),base.permute(0,2,3,1).tanh(),bands),dim=-1)
    35	        correction=self.late_head(design).unsqueeze(1)
    36	        return base+torch.where(positive.bool(),correction,0.),updated
```

## 冻结保护头的增益范围：NativeWeakEchoVerifier.py

本机源码：`standalone/NativeWeakEchoVerifier.py`；原行号：10–31；原文件 SHA-256：`280e1797c8a38e23cd75240b4df526b3ec697cede2d93ed3285a47951fc7a8ce`。

```text
    10	    """Soft bypass for strong corrections. No dependence on future or labels."""
    11	    return (-(old/.22).square()).exp()
    12
    13
    14	class WeakEchoVerifier(nn.Module):
    15	    def __init__(self):
    16	        super().__init__()
    17	        self.parent=AmplitudeHead().requires_grad_(False)
    18	        self.head=nn.Sequential(nn.Linear(BASE+EXTRA+1,24),nn.SiLU(),nn.Linear(24,1))
    19	        nn.init.zeros_(self.head[-1].weight)
    20	        nn.init.constant_(self.head[-1].bias,2.2)
    21	        self.register_buffer('mix',torch.ones(()))
    22
    23	    def retain_logit(self,x,cue,gain):
    24	        return self.head(torch.cat((x,cue,gain.tanh()[...,None]),-1)).squeeze(-1)
    25
    26	    def from_parent(self,gain,score):
    27	        return gain*(1.-self.mix*(1.-score.sigmoid())*attenuation(gain))
    28
    29	    def forward(self,x,cue,old):
    30	        gain=self.parent(x,cue,old)
    31	        return self.from_parent(gain,self.retain_logit(x,cue,gain))
```

## 板端绑定、CUDA Graph、计时和实采校验范围：RunQiandaoRecordedSmallBoardPilot.py

本机源码：`standalone/RunQiandaoRecordedSmallBoardPilot.py`；原行号：56–113、128–152、225–235、248–278、341–350；原文件 SHA-256：`fed90e744eca7f10f6eb319dacf7706826783414eda3adaa167069bf5bbe61ad`。

```text
    56	class Runtime:
    57	    def __init__(self,path,*,bins=N):
    58	        import torch
    59	        import tensorrt as trt
    60	        if bins not in (12189,12721):
    61	            raise ValueError('Declare one of the two exported native geometries.')
    62	        padded=bins+(-bins)%8
    63	        state=tuple((2,128*(i+1),1,padded//2**i) for i in range(4))
    64	        shapes=dict(zip((*INPUTS,*OUTPUTS),((16,28,1,padded),(16,1,1,padded),(16,1,bins,6),*state,
    65	                      (16,bins),*state),strict=True))
    66	        self.log=trt.Logger(trt.Logger.ERROR)
    67	        self.rt=trt.Runtime(self.log)
    68	        self.eng=self.rt.deserialize_cuda_engine(path.read_bytes())
    69	        if self.eng is None:
    70	            raise ValueError('Exact target-local FP32 plan did not deserialize.')
    71	        self.ctx=self.eng.create_execution_context()
    72	        if self.ctx is None:
    73	            raise ValueError('Could not create TensorRT execution context.')
    74	        names={self.eng.get_tensor_name(i) for i in range(self.eng.num_io_tensors)}
    75	        if names!=set(shapes):
    76	            raise ValueError('Unexpected engine inputs or outputs.')
    77	        self.mem={}
    78	        for n,s in shapes.items():
    79	            inbound=n in INPUTS
    80	            if (self.eng.get_tensor_dtype(n)!=trt.float32 or tuple(self.eng.get_tensor_shape(n))!=s
    81	                    or (self.eng.get_tensor_mode(n)==trt.TensorIOMode.INPUT)!=inbound):
    82	                raise ValueError('Do not load the v42 engine, another precision, or a resized model.')
    83	            a=torch.zeros(s,dtype=torch.float32,device='cuda:0')
    84	            if not self.ctx.set_tensor_address(n,a.data_ptr()):
    85	                raise ValueError('Engine device binding failed.')
    86	            self.mem[n]=a
    87	        self.capture_stream=torch.cuda.Stream()
    88	        now=torch.cuda.current_stream()
    89	        self.capture_stream.wait_stream(now)
    90	        with torch.cuda.stream(self.capture_stream):
    91	            self._step()
    92	        now.wait_stream(self.capture_stream)
    93	        self.graph=torch.cuda.CUDAGraph()
    94	        with torch.cuda.graph(self.graph,stream=self.capture_stream):
    95	            self._step()
    96	        self.reset()
    97
    98	    def _step(self):
    99	        import torch
   100	        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
   101	            raise RuntimeError('TensorRT enqueue failed; no PyTorch fallback.')
   102	        for a,b in zip(INPUTS[3:],OUTPUTS[1:],strict=True):
   103	            self.mem[a].copy_(self.mem[b])
   104
   105	    def reset(self):
   106	        for n in INPUTS[3:]:
   107	            self.mem[n].zero_()
   108
   109	    def run(self,f,m,c):
   110	        for n,v in zip(INPUTS[:3],(f,m,c),strict=True):
   111	            self.mem[n].copy_(v)
   112	        self.graph.replay()
   113	        return self.mem['gain']
   128	        from NativeCrossfitBeamCue import projection_compute,side_flanks
   129	        from NativeLocalSpectralCue import _local_spectra_compute
   130	        from NativePairedComplexInnovation import coordinates,project_planes
   131	        from torch.nn import functional as F
   132	        n=self.n
   133	        z=torch.from_numpy(np.ascontiguousarray(iq)).cuda().reshape(2,8,1,n)
   134	        f,m=self.stream._step_host_checked_complex64(z.reshape(16,n))
   135	        mb=m[...,:n].bool().reshape_as(z)
   136	        s=torch.cat([_local_spectra_compute(z[k],mb[k],self.hann).flatten(-2).permute(0,3,1,2)
   137	                     for k in (0,1)],0)
   138	        f=F.pad(torch.cat((f[...,:n],s),1),(0,self.p-n,0,0),mode='replicate')
   139	        cue=[]
   140	        for k in (0,1):
   141	            r,ok=z[k],mb[k]
   142	            p,c,d=projection_compute(r,ok)
   143	            local=side_flanks(r.abs().square())/side_flanks(ok.float()).clamp_min(1.)
   144	            _,_,g,p=coordinates(r,p,local)
   145	            cue.append(project_planes(g,torch.view_as_real(p),c,d))
   146	        cue=torch.cat(cue,dim=0)
   147	        start,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
   148	        start.record()
   149	        gains=self.net.run(f,m,cue)
   150	        end.record()
   151	        out=gains.cpu().numpy().reshape(2,8,n)
   152	        return out,start.elapsed_time(end)
   225	        if legacy is None and (set(h)!=set(('pings','rate','kind','engine_sha256'))
   226	                or h['pings']!=32 or h['kind']!='recorded_native_AB'):
   227	            raise ValueError('Explicit short technical replay only.')
   228	        if legacy is not None:
   229	            legacy.validate_header(h)
   230	        digest=hashbytes(a.engine.read_bytes())
   231	        if digest!=h['engine_sha256']:
   232	            raise ValueError('Wrong board-local engine.')
   233	        net=Runtime(a.engine)
   234	        scope={} if legacy is None else legacy.initialize()
   235	        send(out,dict(kind='ready',engine_sha256=digest,platform='NX_FP32_CUDA_Graph',native=[2,8,N],**scope))
   248	            if not np.isfinite(iq).all() or not np.isfinite(support).all() or np.any((support<0)|(support>1)):
   249	                raise ValueError('Invalid raw observations or observed geometry mask.')
   250	            t0=time.monotonic()
   251	            if reset:
   252	                stream=NativeLearnedGain(net,N)
   253	            gains,core=stream.run(iq)
   254	            seconds=time.monotonic()-t0
   255	            obs=iq[:,:,0]!=0
   256	            if not np.isfinite(gains).all() or np.any(gains[~obs]):
   257	                raise ValueError('Protect original missing values; reject nonfinite.')
   258	            amp=np.abs(iq[:,:,0].astype(np.complex128))
   259	            o=amp*np.exp(gains.astype(np.float64)*support)
   260	            if np.any((o!=amp)&((support==0)|(~obs))):
   261	                raise ValueError('Water column changed.')
   262	            if legacy is None:
   263	                shrink=lambda a:cv2.resize(a.reshape(16,N).astype(np.float32),(512,16),interpolation=cv2.INTER_AREA).reshape(2,8,512)
   264	                raw.append(np.sqrt(shrink(amp**2)));denoise.append(np.sqrt(shrink(o**2)))
   265	                masks.append(shrink((support>.5)&obs)>.6)
   266	                nav_s=None
   267	            else:
   268	                if meta['record_index']!=i:
   269	                    raise ValueError('Unexpected original record index.')
   270	                mask=(support>.5)&obs
   271	                nav_s=(legacy.add_received(i,meta['t'],amp,o,mask,meta) if hasattr(legacy,'add_received')
   272	                       else legacy.add(i,meta['t'],amp,o,mask))
   273	            extra=legacy.ack_fields(i) if hasattr(legacy,'ack_fields') else {}
   274	            send(out,dict(kind='ack',seq=i,t=meta['t'],body_sha256=meta['body_sha256'],
   275	                corrected_values=int((o!=amp).sum()),native_values=amp.size,
   276	                model_ms=core,full_denoise_ms=1000*seconds,validation_gain_bytes=GAIN_BYTES if i<3 else 0,
   277	                **({} if legacy is None else dict(navigation_ms=1000*nav_s)),**extra))
   278	            if i<3:
   341	            if (ack['kind']!='ack' or ack['seq']!=i or ack['t']!=ts or ack['body_sha256']!=key
   342	                    or ack['native_values']!=16*N or ack['validation_gain_bytes']!=(GAIN_BYTES if i<3 else 0)):
   343	                raise ValueError('Loss, duplication, or reordered real inputs.')
   344	            if i<3:
   345	                actual=np.frombuffer(readn(child.stdout,GAIN_BYTES),dtype='<f4').reshape(2,8,N)
   346	                maxerr=float(np.max(np.abs(actual-gpu[i])))
   347	                ack['original_fp32_gain_max_difference']=maxerr
   348	                if not np.isfinite(maxerr) or maxerr>2.e-4:
   349	                    raise ValueError('Board gain precision differs from prior real-ping FP32.')
   350	            ack['host_lateness_ms']=1000*(first-started-due)
```

## H0 固定引擎、实际加载校验和恢复：RunQiandaoHistoricalClockAB.py

本机源码：`standalone/RunQiandaoHistoricalClockAB.py`；原行号：21–29、84–92、153–162；原文件 SHA-256：`7bec4fdf69a946b6c0216187248157da97b12c007db830f46368beb527490a97`。

```text
    21	PROTOCOL = "qiandao_historical_native_clock_matched_ab_guarded_v2"
    22	GEOMETRY = "SideScanLoopGeometry.py"
    23	ARCHIVED_RUNTIME_FILES = (*FILES, GEOMETRY)
    24	GEOMETRY_SHA = "df24659604f31e7ec42a9d3280b4ed62fe30adb05fed119ee239aa43a3db7267"
    25	INTERRUPTED_TRACE_SHA = "46103a3fec13dc2dac584dd1b233acbe6663e2461190fc4decac76e765287fdc"
    26	INTERRUPTED_TRACE_PINGS = 19499
    27	TOTAL = 62458
    28	ARMS = ("raw", "student")
    29	ENGINE = "b20daf8809e304ab9a8f2032c391284b3ef1c837277dd4f2b8bd876135603bbf"
    84	        rt = a.legacy_root / "runtime"
    85	        source = rt / "standalone"
    86	        digests = dict(code=hashes(source, ARCHIVED_RUNTIME_FILES), prior=hashes(a.prior, PRIOR),
    87	                       engine=file_sha(a.engine), loop_observation_guard=file_sha(Path(loop_guard.__file__)))
    88	        require(h["kind"] == PROTOCOL and h["digests"] == digests
    89	                and 1 <= h["pings"] <= TOTAL and digests["prior"]["backscatter.npy"] == OLD_PRIOR
    90	                and digests["engine"] == ENGINE and digests["code"][GEOMETRY] == GEOMETRY_SHA
    91	                and file_sha(a.legacy_root / "loftr_outdoor.ckpt") == WEIGHT_SHA,
    92	                "Unverified board, historical navigator, engine or bottom map.")
   153	            apply_motion(meta["motion"], stamp)
   154	            last_ping = stamp
   155	            z = np.frombuffer(data, dtype="<c8").reshape(2, 8, BINS)
   156	            gate = observed_water_support(z, receiver.parameters.thermal_noise_rms)
   157	            gain, model_ms = learned.run(z)
   158	            restored = z * np.exp(gain * gate)
   159	            require(np.isfinite(restored).all() and np.all(restored[gate == 0] == z[gate == 0]),
   160	                    "Invalid restoration or changed protected water column.")
   161	            ack = dict(kind="ack", i=i, t=stamp, sha=sha, model_ms=model_ms,
   162	                       corrected=int(np.count_nonzero(restored != z)))
```

## H5 整个板端去噪段的计时边界：RunQiandaoNoisyCombBoardAB.py

本机源码：`standalone/RunQiandaoNoisyCombBoardAB.py`；原行号：95–109；原文件 SHA-256：`4f9f42fd87bf439a104d0e6f35c8c7971a00350eddbc3b54f0519e4ec8bbd13a`。

```text
    95	                prev=item
    96	            z=np.frombuffer(raw,dtype='<c8').reshape(2,8,BINS)
    97	            if not np.isfinite(z).all():raise ValueError('Finite I/Q required.')
    98	            from OnlineSonarDenoising import observed_water_support
    99	            t0=time.monotonic()
   100	            gate=observed_water_support(z,receiver.parameters.thermal_noise_rms)
   101	            gain, model_ms=learned.run(z)
   102	            if not np.isfinite(gain).all() or np.any(gain[z==0]):
   103	                raise ValueError('Invalid learned gain.')
   104	            z2=z*np.exp(gain*gate)
   105	            if not np.isfinite(z2).all() or np.any(z2[gate==0]!=z[gate==0]):
   106	                raise ValueError('Protected receive-water samples changed.')
   107	            info=dict(kind='ack',seq=i,sha=digest, model_ms=model_ms,
   108	                      denoise_ms=1000*(time.monotonic()-t0),corrected=int(np.count_nonzero(z2!=z)))
   109	            for k,v in zip(KEYS,(z,z2),strict=True):
```

## past4 + current、有效域与真实距离相位：CausalSonarFeatures.py

本机源码：`standalone/CausalSonarFeatures.py`；原行号：48–85；原文件 SHA-256：`08e2e509a565eff3696f321938e7cb2350f2ffbbf31cdbf21965b609f0f6aad7`。

```text
    48	                or (self.device.index is not None and observed.device != self.device)
    49	                or self.domain < 2):
    50	            raise ValueError('The host-validated path accepts only transferred complex64 IQ.')
    51	        return self._step(observed, value_checked=True)
    52
    53	    def _step(self, observed, *, value_checked):
    54	        if (observed.ndim != 2 or (self.domain >= 2) != observed.is_complex()
    55	                or (self.domain < 2 and (observed < 0).any())
    56	                or (not value_checked and not torch.isfinite(observed).all())
    57	                or (self.shape is not None and tuple(observed.shape) != self.shape)):
    58	            raise ValueError('Aligned actual observations matching the explicit measurement domain required.')
    59	        amplitude = observed.abs()
    60	        mask = amplitude > 0
    61	        safe = torch.where(mask, amplitude, torch.ones_like(amplitude))
    62	        q = torch.where(mask, safe.log(), 0).float()
    63	        phase = torch.zeros(observed.shape[0], 2, observed.shape[1], device=self.device)
    64	        if self.domain >= 2:
    65	            unit = torch.where(mask, observed / safe, 0)
    66	            gradient = unit[:, 1:] * unit[:, :-1].conj()
    67	            valid = mask[:, 1:] & mask[:, :-1]
    68	            phase = F.pad(torch.stack((gradient.real, gradient.imag), dim=1) * valid[:, None], (1, 0))
    69	        positive = mask[:, None, None].float()
    70	        if self.cache_range_context:
    71	            # Each received row's range-only pooling is independent of every
    72	            # other ping. Cache those exact intermediates, not a running sum:
    73	            # the five-row temporal reduction retains its original order.
    74	            current = q[:, None, None]
    75	            self.range_totals.append(F.avg_pool2d(current * positive, (1,65), stride=1, padding=(0,32)))
    76	            self.range_weights.append(F.avg_pool2d(positive, (1,65), stride=1, padding=(0,32)))
    77	            self.range_totals = self.range_totals[-5:]
    78	            self.range_weights = self.range_weights[-5:]
    79	            missing = 5-len(self.range_totals)
    80	            total = F.avg_pool2d(F.pad(torch.cat(self.range_totals,dim=2),(0,0,missing,0)),(5,1),stride=1)
    81	            weight = F.avg_pool2d(F.pad(torch.cat(self.range_weights,dim=2),(0,0,missing,0)),(5,1),stride=1)
    82	            mean = total / torch.where(weight>0,weight,torch.ones_like(weight))
    83	            features = torch.cat(((current-mean)*positive,phase[:,:,None]*positive),dim=1)
    84	        else:
    85	            self.logs.append(q[:, None])
```

## 32 点 FFT 与 24 谱特征：NativeLocalSpectralCue.py

本机源码：`standalone/NativeLocalSpectralCue.py`；原行号：7–12、28–47；原文件 SHA-256：`aa6ebeeb6ac1b213b7ac2dd915ed56a9f129ec187a91e574844e57c79816eb9b`。

```text
     7
     8	WINDOW=32
     9	BANDS=8
    10	FIELDS=3
    11	EXTRA=BANDS*FIELDS
    12	HANN=tuple(.5-.5*math.cos(2*math.pi*j/(WINDOW-1)) for j in range(WINDOW))
    28	def _local_spectra_compute(iq,valid,win):
    29	    """Tensor-only body. Callers are responsible for validated shapes/support."""
    30	    scale=iq.abs().amax(-1,keepdim=True)
    31	    z=iq/torch.where(scale>0,scale,1.)
    32	    pad=(WINDOW//2-1,WINDOW//2)
    33	    ff=F.pad(z,pad).unfold(-1,WINDOW,1)
    34	    count=F.pad(valid,pad,value=False).unfold(-1,WINDOW,1).sum(-1)
    35	    available=valid & (count>=WINDOW//2)
    36	    s=torch.fft.fft(ff*win,dim=-1)
    37	    raw=(s.real.square()+s.imag.square()).unflatten(-1,(BANDS,WINDOW//BANDS)).sum(-1)
    38	    energy=raw.sum(-1,keepdim=True)
    39	    available=available & (energy.squeeze(-1)>0)
    40	    normalized=(raw/torch.where(available[...,None],energy,1.))*available[...,None]
    41	    peers=available.sum(0,keepdim=True)-available.to(torch.int64)
    42	    denom=peers.clamp_min(1)[...,None]
    43	    average=(normalized.sum(0,keepdim=True)-normalized)/denom
    44	    square=normalized.square()
    45	    var=((square.sum(0,keepdim=True)-square)/denom-average.square()).clamp_min(0.)
    46	    group=available[...,None] & (peers>=2)[...,None]
    47	    return torch.stack((normalized,average*group,var.sqrt()*group),dim=-1)
```

## 实际选择的骨干试验配置及教师训练权重：TrainNativeSignedResponse172.py

本机源码：`standalone/TrainNativeSignedResponse172.py`；原行号：28–37、111–119、152–171；原文件 SHA-256：`fd8af929b8f9dada3453befba1cf11d47775a8053f5defd50fc7af9312531ea2`。

```text
    28	from TrainJointBeamNativeLogTailDevelopment import idle_gpu_unit
    29
    30
    31	ARMS=('control','signed_response')
    32	OFFSET=92000
    33	POINTS=(0,50,200)
    34	STEP=1000
    35	AUDIT='e6e7576bca034ae75f679c3edca9f823f85abdf1904319e051dd643ac86d3766'
    36	GAIN_PROOF='8e00fc10fa9e5f0f4f09673786aa7c296d60c6bc80a078212022b72122eba25c'
    37	PATHS=('parent','snapshot','audit','gain_proof','source_archive','admission','profile','acceptance','offline_teacher','out')
   111	        teacher.to(device)
   112	    groups,sources,profile=BeamGroups(records),Sources(),read(a.profile)
   113	    a.sequence,a.burn_in,a.truncation,a.paired_mode=32,32,8,DIFFERENCE
   114	    dist.init_process_group('nccl',timeout=timedelta(minutes=20))
   115	    sync=lambda:dist.barrier(device_ids=[local])
   116	    if not rank:a.out.mkdir(mode=0o700,exist_ok=False)
   117	    sync()
   118	    nets=[DistributedDataParallel(m,device_ids=[local],broadcast_buffers=False) for m in models]
   119	    opts=[torch.optim.AdamW(m.parameters(),lr=3e-5,weight_decay=1e-5) for m in models]
   152	    try:
   153	        evaluate(0)
   154	        ledger=a.out/f'rank_{rank}.jsonl'
   155	        with ledger.open('x',buffering=1) as stream:
   156	            for step in range(1,POINTS[-1]+1):
   157	                draw=OFFSET+4*(step-1)+rank
   158	                anchor=plan_draw(records,draw,curriculum=PUBLIC_HEAVY)
   159	                noisy,bare,core,meta=paired_example(sources,groups,anchor,4000000+draw,device,profile)
   160	                native=anchor['role']=='native_train'
   161	                reading=[step_update(m,net,opt,noisy,bare,core,None,meta,a,4,teacher_weight=0.,
   162	                          prepare_features=student_tensors_with_baseline,native_signed_response=native and i==1)
   163	                         for i,(m,net,opt) in enumerate(zip(models,nets,opts))]
   164	                x,y=reading
   165	                if (any(x[k]!=y[k] for k in ('score_rows','observed_warmup_rows','slot_scored_values','actual_scored_added_log_samples'))
   166	                        or x['teacher_distillation'] or y['teacher_distillation']
   167	                        or (y.get('signed_response_protocol')==PROTOCOL)!=native
   168	                        or not math.isclose(y['loss'],y['restoration']+y['identity_loss']+y.get('signed_response_added_loss',0.),
   169	                                            rel_tol=5e-7,abs_tol=5e-9)):
   170	                    raise ValueError('Only the declared native restoration addition may differ between paired arms.')
   171	                for lr in schedules:lr.step()
```

## 频谱头三臂及优化器：TrainLateSpectralRangeBalance172.py

本机源码：`standalone/TrainLateSpectralRangeBalance172.py`；原行号：22–26、58–74；原文件 SHA-256：`f912e0bf35063d6b02f5b46b54ac409b1d661208a7da1c383e0865f560f89f45`。

```text
    22	PROTOCOL='late_native_same_noise_range_balanced_identity_DEV_ONLY_v1'
    23	ARMS=('nominal_identity','range_balanced_identity','range_and_bright_tail')
    24	STEPS=128
    25	DIAGNOSIS='452ab46e77a709238da57672cbefdbcddc07c4e977f3ef7caab4b50fe4c64c2b'
    26
    58	def fit(initial,cohorts,device,target,steps=STEPS):
    59	    models=[copy.deepcopy(initial) for _ in ARMS];weights=[[16.,16.] for _ in models]
    60	    opts=[torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1.e-5) for m in models]
    61	    rng=torch.Generator(device=device).manual_seed(325621)
    62	    track=[]
    63	    for i in range(steps):
    64	        cohort=cohorts[i%len(cohorts)]
    65	        sampled=[]
    66	        for noisy,normal,balanced in cohort:
    67	            if len(normal['target'])!=len(balanced['target']):
    68	                raise ValueError('Both identity conditions must have the same count.')
    69	            sampled.append((torch.randint(len(noisy['target']),(BATCH,),device=device,generator=rng),
    70	                            torch.randint(len(normal['target']),(BATCH,),device=device,generator=rng)))
    71	        stage={}
    72	        for arm,(m,o) in enumerate(zip(models,opts)):
    73	            results=[];errors=[]
    74	            o.zero_grad(set_to_none=True)
```

## 保护头原臂与独立候选试验：TrialNativeSplitFlankProjection172.py

本机源码：`standalone/TrialNativeSplitFlankProjection172.py`；原行号：26–29、129–166；原文件 SHA-256：`b9b060f8dab1fb8484fdec7dc31a7d3f1487ca5b4b0e40685c2f929a9c0bb781`。

```text
    26	ARMS=('weak_verifier_original','merged_flanks','independent_flanks')
    27	STEPS=384
    28	BACKGROUND='ba90d6775ae37bc3f8141d3fdecdf4dae6cacedceb6b78652ddba927da8f3317'
    29
   129	            g=guard_eval(held,m,k,s)
   130	            tries.append(dict(scale=s,guard=g,score=rank(g,base,goal)))
   131	        ok=[x for x in tries if x['score'] is not None]
   132	        s=min(ok,key=lambda q:q['score'])['scale'] if ok else None
   133	        m.mix.fill_(1. if s is None else s)
   134	        trace.append(dict(arm=name,train_guard_selected_mix=s,tried=tries))
   135	    return [seed.frozen,*heads],trace
   136
   137
   138	def main(a):
   139	    z,g,p=a.previous/'summary.json',a.gate_run/'summary.json',a.parent_run/'summary.json'
   140	    zz,gg,pp,bb=read(z),read(g),read(p),read(a.expanded_plan)
   141	    if (sha(z)!=BACKGROUND or sha(g)!=GATED or sha(p)!=PARENT or sha(a.expanded_plan)!=PLAN
   142	            or any(q['final_holdout_consumed'] for q in (zz,gg,pp,bb)) or zz['guard_plan']!=gg['guard_plan']
   143	            or len(bb['paired_cohorts'])!=26):
   144	        raise ValueError('Require the previously finished protected matched experiments.')
   145	    w,=[v for v in gg['weights'] if v['method']=='gated_aligned_pair']
   146	    x=a.gate_run/w['file']
   147	    if sha(x)!=w['SHA256']:
   148	        raise ValueError('The radiometrically protected starting weight cannot change.')
   149	    init=torch.load(x,map_location='cpu',weights_only=True)['model']
   150	    evidence=dict(same_ping_only=True,training_labels_use_actual_added_iq=True,received_signal_not_modified=True,
   151	                  fit_windows=52,guard_windows=24,left_right_guard_bins=GUARD,radius_bins=RADIUS)
   152	    def cohorts(records,logs):
   153	        u=plan(records,logs)
   154	        if any(u[k]!=bb[k] for k in u):
   155	            raise ValueError('Original paired source selection changed.')
   156	        a,b=validation_plan(records,logs,bb)
   157	        if b!=gg['guard_plan']:
   158	            raise ValueError('Same disjoint calibration required.')
   159	        evidence['guard_plan']=b
   160	        return bb['paired_cohorts']+a
   161	    run(a,arms=ARMS,cue_fields=EXTRA+FIELDS,control_fields=BASE,steps=STEPS,
   162	        protocol='native_split_range_flanks_complex_estimation_development_v1',
   163	        maker=make_case,pair_preparer=prepare_pair,predictor=predict,trainer=partial(fit,initializer=init),
   164	        trainer_progress=True,cohort_provider=cohorts,evidence=evidence,prepare_progress_every=4)
   165
   166
```
