# 真实接口和冻结导航实现：源码摘录

左列为原行号，间断位置有意未摘录；完整源文件摘要用于对齐原归档。`F/` 代表冻结导航运行时，校验范围及与每次板端握手的区别见[数据一致性](../DATA_CONSISTENCY.md)。这些是审阅材料，不是完整可安装源码。去噪器和 H0/H5 调用参见[第一阶段原行号证据](../../sonar_denoising_model_audit/evidence/SOURCE_EXCERPTS.md)。

## 实采幅度转成导航包

`standalone/RunQiandaoRecordedLegacyNavAB.py`，44–59；SHA-256：`ec779a875db55d90556cedf8be21218c4f8f76efdd30cdb9e5f79ef5f57d52da`。

```text
    44	def received_packet(stamp, amplitude, valid, packet_type):
    45	    import cv2
    46	    # Same receiver area-power binning as the old HIL. This is only its
    47	    # internal feature input, not a resized native product or model input.
    48	    raster = lambda x: cv2.resize(x.reshape(16, N), (512, 16), interpolation=cv2.INTER_AREA).reshape(2, 8, 512)
    49	    good = raster(valid.astype(np.float64)) > .6
    50	    amp = np.sqrt(raster(np.where(valid, amplitude**2, 0.)))
    51	    good &= amp > max(float(amp.max())*1e-6, np.finfo(float).tiny)
    52	    side, row, index = np.nonzero(good)
    53	    dx = 1480. / (2 * 37650.)
    54	    full_range = N*dx
    55	    # Native sample k (1-based) is at k*dx. Centres of the receiver's
    56	    # 512 equal-width bins retain its original physical scale and gaps.
    57	    return packet_type(stamp, (index+.5)*full_range/512., amp[good], row, side,
    58	                       (np.arange(8)-3.5)*.15, index, full_range)
    59
```

## 断点只重建去噪历史，奇数观察交给两个导航实例

`standalone/RunQiandaoRecordedTrackAB.py`，46–87；SHA-256：`0fc5dd3a8ee8cdcbeb4a69b15f8873a3917c64f4bf5b3941bb7c5ab7b15e99c8`。

```text
    46	    def expect_reset(self,index):
    47	        return index in self.breaks
    48
    49	    def add_received(self,index,stamp,raw,student,common_support,meta):
    50	        if set(meta)!={'seq','record_index','t','reset','body_sha256','motion'}:
    51	            raise ValueError('Only noisy actual I/Q plus non-position motion proxy events are permitted.')
    52	        if index%2==0:
    53	            if meta['motion']:raise ValueError('Map acquisitions may not reach the navigator.')
    54	            self.last_query_record=None
    55	            return 0.
    56	        begin=time.monotonic()
    57	        for typ,ts,value in meta['motion']:
    58	            if (typ not in (0,1,2) or (ts,typ)<=self.motion_prev or ts>stamp or
    59	                    not np.isfinite(value) or abs(value)>20.):
    60	                raise ValueError('Non-position motion proxy stream changed order/schema.')
    61	            self.motion_prev=(ts,typ)
    62	            for nav in self.branches.values():
    63	                if typ==0:nav.imu(ts,np.array([0.,0.,value]),self.force)
    64	                elif typ==1:nav.dvl(np.array([value,0.,0.]))
    65	                elif value==DEPTH:nav.depth(DEPTH)
    66	                else:raise ValueError('No calibrated pressure samples were provided.')
    67	        row=dict(record_index=index,timestamp_s=stamp)
    68	        for key,amp in (('raw',raw),('student',student)):
    69	            nav=self.branches[key]
    70	            result=nav.side_scan(received_packet(stamp,amp,common_support,self.packet_type))
    71	            row[key]=dict(positions={n:result[n].tolist() for n in ('DR','Local','Global')},
    72	                          frontend=nav.last_feature,
    73	                          map_epoch=nav.last_backend if nav.observation_id%150==0 else None)
    74	        if row['raw']['positions']['DR']!=row['student']['positions']['DR']:
    75	            raise ValueError('Motion input must be identical in both arms.')
    76	        self.last_query_record=self.sanitize(row)[0]
    77	        self.rows.append(index)
    78	        return time.monotonic()-begin
    79
    80	    def ack_fields(self,i):
    81	        return dict(query=self.last_query_record)
    82
    83	    def finish(self):
    84	        if self.rows!=list(range(1,COUNT,2)):
    85	            raise ValueError('Incomplete query IDs or overlap with reference IDs.')
    86	        return self.sanitize(dict(query_pings=len(self.rows),
    87	            diagnostics={k:n.diagnostics() for k,n in self.branches.items()},state_resets=len(self.breaks)))[0]
```

## 仿真脉压输出转 512 格包（非旧去噪类）

`standalone/SideScanIQHistory.py`，69–88；SHA-256：`bfa0e9f4aff127f2a9952b8f64097a8da9ccdf5cda0f5056244dbe0ff6dc4c5c`。

```text
    69	def packet_from_compressed_IQ(timestamp, compressed_IQ, parameters, *, bins=512):
    70	    """Observed envelopes only; no world coordinates, clean IQ or height mask.
    71
    72	    A numerical floor rejects FFT roundoff. The declared receiver thermal
    73	    floor rejects pure receiver-noise bins. Neither uses the clean signal.
    74	    """
    75	    import cv2
    76	    z = np.asarray(compressed_IQ)
    77	    if z.shape != (2, 8, parameters.range_bins) or not np.iscomplexobj(z) or bins < 2:
    78	        raise ValueError('Native 2x8-beam complex pulse-compressed observation required.')
    79	    power = abs(z).astype(float)**2
    80	    amplitude = np.sqrt(cv2.resize(power.reshape(16, -1), (bins, 16), interpolation=cv2.INTER_AREA)).reshape(2, 8, bins)
    81	    numerical_floor = max(float(amplitude.max())*1e-6, np.finfo(float).tiny)
    82	    detection_floor = max(numerical_floor, 1.8*parameters.thermal_noise_rms)
    83	    good = amplitude > detection_floor
    84	    side, row, index = np.nonzero(good)
    85	    ranges = (index+.5)/bins*parameters.max_range_m
    86	    return SideScanFeaturePacket(timestamp=float(timestamp), ranges_m=ranges,
    87	        amplitudes=amplitude[good], rows=row, sides=side, sample_sequence=index,
    88	        row_centres_m=parameters.navigation_row_centres_m, max_range_m=parameters.max_range_m)
```

## 冻结接口类型、束标签与物理尺度

`F/standalone/PureSideScanFeatures.py`，14–42、139–151；SHA-256：`e551d3e54515921d0ebd77e3f86e9936e7ef7c2ff82c191778b8e876298b8255`。

```text
    14	@dataclass(frozen=True)
    15	class SideScanFeaturePacket:
    16	    timestamp: float
    17	    ranges_m: np.ndarray
    18	    amplitudes: np.ndarray
    19	    rows: np.ndarray
    20	    sides: np.ndarray
    21	    row_centres_m: np.ndarray
    22	    sample_sequence: np.ndarray
    23	    max_range_m: float = 250.
    24
    25	    def __post_init__(self):
    26	        r, a, row, side = map(np.asarray, (self.ranges_m, self.amplitudes, self.rows, self.sides))
    27	        centres = np.asarray(self.row_centres_m, float)
    28	        sequence = np.asarray(self.sample_sequence)
    29	        if r.ndim != 1 or any(x.shape != r.shape for x in (a, row, side)):
    30	            raise ValueError('One raw range/amplitude/side/row per side-scan return required.')
    31	        if centres.shape != (8,) or not np.all(np.diff(centres) > 0):
    32	            raise ValueError('Eight calibrated simultaneous phase centres required.')
    33	        if sequence.shape != r.shape or np.any(sequence < 0) or not np.equal(sequence, np.rint(sequence)).all():
    34	            raise ValueError('Original nonnegative echo acquisition indices required to preserve missing-data gaps.')
    35	        if (not np.isfinite(r).all() or not np.isfinite(a).all() or np.any(r <= 0)
    36	                or np.any(a <= 0) or not np.isin(row, range(8)).all() or not np.isin(side, [0, 1]).all()):
    37	            raise ValueError('Only valid measured side-scan scalars and labels are accepted.')
    38	        for name, values, dtype in [('ranges_m', r, float), ('amplitudes', a, float),
    39	                ('rows', row, np.int64), ('sides', side, np.int64), ('row_centres_m', centres, float),
    40	                ('sample_sequence', sequence, np.int64)]:
    41	            copied = np.array(values, dtype=dtype, copy=True); copied.flags.writeable = False
    42	            object.__setattr__(self, name, copied)
   139	def image_from_packet(packet, bins=512, scale=4):
   140	    rb = np.minimum((packet.ranges_m/packet.max_range_m*bins).astype(int), bins-1)
   141	    col = np.where(packet.sides == 0, bins-1-rb, bins+rb)
   142	    image = np.zeros((8, 2*bins), float)
   143	    np.maximum.at(image, (packet.rows, col), packet.amplitudes)
   144	    u8, mask = normalize_feature_image(image, image > 0, scale)
   145	    pixels, descriptors = extract_features(u8, mask)
   146	    # Pixel-centre inverse of OpenCV resize. No nearest physical row or XYZ.
   147	    x = packet.row_centres_m[0]+((pixels[:, 1]+.5)/scale-.5)*.15
   148	    r = np.where(pixels[:, 0] < bins, bins-.5-pixels[:, 0],
   149	                 -(pixels[:, 0]-bins+.5))*packet.max_range_m/bins
   150	    return dict(image=u8, pixels=pixels, descriptors=descriptors, along_m=x,
   151	                signed_slant_m=r, bins=bins, timestamp=packet.timestamp)
```

## 同一导航入口及地图、回环、斜距和输出

`F/standalone/PureSideScanFeatureNavigation.py`，205–245、273–315；SHA-256：`7e0ecb63139da121c0d03fa9b545ef9a41c89994ff346a0b7eb91aa17307a6da`。

```text
   205	    def side_scan(self, packet):
   206	        from oceansim.slam.global_tracking import GlobalPoseEvidence, ParticleLikelihoodObservation
   207	        # Original frozen engine propagates held IMU to every sonar stamp,
   208	        # including stamps falling between the 50 Hz IMU events.
   209	        if self.local.state.timestamp < packet.timestamp-1e-12:
   210	            for f in (self.local, self.dr):
   211	                f.propagate(packet.timestamp, self.latest_gyro, self.latest_acceleration)
   212	        self.frontend(packet)
   213	        raw = state_pose2(self.local.state)
   214	        self.observation_id += 1
   215	        self.window.add(packet, raw, self.observation_id, vertical_position_m=self.local.state.position[2])
   216	        self.ensure_keyframe(packet.timestamp, raw)
   217	        target = self.raw_graph_pose(raw)
   218	        if self.observation_id % 150 == 0:
   219	            self.map_attempts += 1
   220	            # Observation-only diagnostics: retain periodic/turn examples
   221	            # using ESTIMATED history, never programmed route labels/truth.
   222	            historical_yaw = np.unwrap([entry[1][2] for entry in self.window.scans])
   223	            turning_window = float(np.ptp(historical_yaw)) > np.deg2rad(15.)
   224	            sample_debug = self.observation_id % 1500 == 0 or turning_window
   225	            try:
   226	                # Use projection only when the legacy strip would discard or
   227	                # reverse acquired scan directions. This is representation
   228	                # support, never a true-pose/error-based acceptance threshold.
   229	                needs_projection = (self.submap_representation == 'adaptive_projected'
   230	                    and np.any(np.cos(historical_yaw-historical_yaw[-1]) < .5))
   231	                if needs_projection:
   232	                    from SideScanProjectedSubmap import build_projected_submap
   233	                    from SideScanProjectedMatching import match_projected_prior
   234	                    self.projected_map_attempts += 1
   235	                    query = build_projected_submap(self.window, raw)
   236	                    matched = match_projected_prior(self.map, query, target, self.image_matcher)
   237	                    self.projected_map_accepts += int(matched['accepted'])
   238	                else:
   239	                    query = self.window.build(raw)
   240	                    matched = match_feature_prior(self.map, query, target, image_matcher=self.image_matcher,
   241	                                                   registration_mode=self.scan_plane_registration)
   242	                if len(self.backend_query_examples) < 3 or (sample_debug and len(self.backend_query_examples) < 32):
   243	                    self.backend_query_examples.append(dict(query, seed_pose2=target.copy()))
   244	            except ValueError as exc:
   245	                matched = dict(accepted=False, reason=str(exc), matches=0)
   273	                query_features=matched.get('query_feature_count'), map_features=matched.get('map_feature_count'),
   274	                observable_rank=matched.get('observable_rank'),
   275	                submap_assembly=None if query is None else json.loads(query['assembly_json']),
   276	                status=decision.status, null_probability=decision.null_probability,
   277	                cohorts=len(decision.cohorts), feature_likelihood_only=True,
   278	                measured_depth_likelihood_terms=0, measured_depth_graph_factors=0,
   279	                DVL_altitude_used=False, optical_flow_used=False)
   280	            cohort = decision.confirmed_cohort if decision.confirmed_with_new_independent_evidence else None
   281	            graph_ready = cohort is not None and cohort.last_measurement_is_graph_ready
   282	            if graph_ready:
   283	                self.ensure_keyframe(packet.timestamp, raw, force=True)
   284	            loop = None
   285	            if query is not None and self.historical_loops is not None:
   286	                loop = self.historical_loops.observe(query, raw, self.graph, self.raw_keyframe_poses, packet.timestamp,
   287	                    shared_current_window=bool(graph_ready))
   288	            if loop is not None:
   289	                # Preserve new absolute anchoring instead of replacing it by
   290	                # a relative revisit to a potentially drifted historical pose.
   291	                # Shared current pixels get HALF information in each factor.
   292	                if graph_ready:
   293	                    add_side_scan_feature_prior(self.graph, len(self.graph.poses)-1, cohort.last_measurement_pose2,
   294	                        cohort.last_measurement_information*loop['current_window_information_weight'],
   295	                        score=matched.get('score', 0))
   296	                    self.graph_feature_priors += 1
   297	                    self.shared_feature_windows += 1
   298	                self.graph.optimize()
   299	            elif graph_ready:
   300	                add_side_scan_feature_prior(self.graph, len(self.graph.poses)-1, cohort.last_measurement_pose2,
   301	                    cohort.last_measurement_information, score=matched.get('score', 0))
   302	                self.graph.optimize(); self.graph_feature_priors += 1
   303	            if graph_ready and self.mapped_ranges is not None:
   304	                try:
   305	                    added = self.mapped_ranges.observe(query, matched, raw, self.graph, self.raw_keyframe_poses)
   306	                    self.last_range_change = dict(added_factors=added, observation_id=self.observation_id)
   307	                    if added: self.graph.optimize()
   308	                except ValueError as exc:
   309	                    self.last_range_change = dict(added_factors=0, reason=str(exc),
   310	                                                  observation_id=self.observation_id)
   311	            target = self.raw_graph_pose(raw)
   312	        published = self.soft.update(packet.timestamp, raw, target)
   313	        position = self.local.state.position.copy(); position[:2] = published[:2]
   314	        return dict(DR=self.dr.state.position.copy(), Local=self.local.state.position.copy(),
   315	                    Global=position, raw_pose2=raw, global_pose2=published)
```

## 主贡献相位中心与距离列

`F/standalone/SideScanProjectedSubmap.py`，29–41、83–102；SHA-256：`e9c1abb8572f16068b46aa0fabf861a731e770eb8d939c85eb21c7f6a5f74b4b`。

```text
    29	def projected_feature_observations(query, pixels):
    30	    """Return real source observations; no image-derived horizontal position."""
    31	    y, x, valid = projected_pixel_indices(query, pixels)
    32	    if not valid.all(): raise ValueError('A feature must refer to actually received supported echoes.')
    33	    phase = query['source_phase_centre'][y, x]
    34	    column = query['source_range_column'][y, x].astype(int)
    35	    bins, max_range = query['bins'], query['max_range_m']
    36	    raw_side = np.where(column < bins, 1., -1.)
    37	    ranges = np.where(raw_side > 0, bins-.5-column, column-bins+.5)*max_range/bins
    38	    return dict(origins=query['phase_origins'][phase], normals=query['phase_normals'][phase],
    39	        ranges=ranges, sides=raw_side*query['phase_side_sign'][phase],
    40	        packet_ids=query['phase_packet_ids'][phase], beam_rows=query['phase_beam_rows'][phase],
    41	        timestamps_s=query['phase_timestamps_s'][phase])
    83	    flat = ri*width+ci; size = height*width
    84	    count = np.bincount(flat, minlength=size).reshape(height, width).astype(np.uint32)
    85	    sums = np.bincount(flat, weights=value, minlength=size).reshape(height, width)
    86	    support = count > 0
    87	    image = np.divide(sums, count, out=np.zeros_like(sums), where=support)
    88	    # Keep a genuine source observation at a blend, rather than averaging
    89	    # incompatible phase centres/normals. Largest received amplitude is a
    90	    # deterministic appearance-dominance proxy, not a correctness oracle.
    91	    order = np.lexsort((np.arange(len(flat)), value, flat))
    92	    sorted_flat = flat[order]
    93	    dominant = order[np.r_[sorted_flat[1:] != sorted_flat[:-1], True]]
    94	    labels = np.full(size, -1, np.int32); range_columns = np.zeros(size, np.int32)
    95	    labels[flat[dominant]] = source_phase[dominant]
    96	    range_columns[flat[dominant]] = source_column[dominant]
    97	    result = dict(assembled_amplitude=image, raw_amplitude=image, observation_count=count,
    98	        support=support, source_phase_centre=labels.reshape(height, width),
    99	        source_range_column=range_columns.reshape(height, width), phase_origins=np.concatenate(origin_table),
   100	        phase_normals=np.concatenate(normal_table), phase_side_sign=np.concatenate(sign_table),
   101	        phase_packet_ids=np.concatenate(id_table), phase_beam_rows=np.concatenate(beam_table),
   102	        phase_timestamps_s=np.concatenate(time_table), anchor_pose2=anchor_pose2.copy(),
```

## 外观关联与历史几何的接口

`F/standalone/SideScanLoopClosure.py`，100–136；SHA-256：`0876e73fa4c83fddff66552b899d9a20e413bcc5ca020d2a5bb3edb5bcb63c6f`。

```text
   100	    def match_pair(self, old, query, seed):
   101	        a = old['query']; reverse = bool(np.cos(seed[2]) < 0)
   102	        typed = 'source_phase_centre' in a or 'source_phase_centre' in query
   103	        tiling = None
   104	        if typed:
   105	            from SideScanHistoricalMatching import match_typed_historical
   106	            pa, pb, confidence, tiling = match_typed_historical(a, query, seed, self.matcher)
   107	        else:
   108	            im = np.flip(query['image'], axis=(0, 1)).copy() if reverse else query['image']
   109	            mask = np.repeat(query['support'], query['scale'], axis=0).astype(np.uint8)*255
   110	            if reverse: mask = np.flip(mask, axis=(0, 1)).copy()
   111	            ma = np.repeat(a['support'], a['scale'], axis=0).astype(np.uint8)*255
   112	            pa, pb, confidence = self.matcher(a['image'], im, ma, mask)
   113	            if reverse:
   114	                pb = np.array([im.shape[1]-1, im.shape[0]-1])-pb
   115	        metadata = dict(typed_source_geometry=typed, historical_tiling=tiling,
   116	            appearance_warp_is_not_a_loop_pose_measurement=True,
   117	            association_consensus_model='appearance_affine_pruning_only' if typed else 'legacy_scan_row_consensus')
   118	        if len(pa) < 24:
   119	            return dict(accepted=False, reason='too_few_historical_feature_matches', matches=len(pa), **metadata)
   120	        oa, ob = feature_observations(a, pa), feature_observations(query, pb)
   121	        if typed:
   122	            from SideScanHistoricalMatching import appearance_consensus
   123	            keep = appearance_consensus(a, query, pa, pb, seed)
   124	        else:
   125	            keep = row_compatible_pairs(oa, ob, seed[2])
   126	        if keep.sum() < 24:
   127	            return dict(accepted=False, reason='no_historical_appearance_consensus' if typed else 'no_historical_row_consensus',
   128	                        matches=len(pa), inliers=int(keep.sum()), **metadata)
   129	        fit = fit_range_plane_loop({k: v[keep] for k, v in oa.items()}, {k: v[keep] for k, v in ob.items()}, seed,
   130	            float(query['anchor_vertical_z_m']-a['anchor_vertical_z_m']), minimum_features=24)
   131	        selected = fit.get('selected_feature_indices', np.arange(int(keep.sum())))
   132	        fit.update(matches=len(pa), row_consensus_matches=int(keep.sum()),
   133	                   inliers=fit.get('features', int(keep.sum())), appearance_reversed_for_matching=reverse,
   134	                   **metadata,
   135	                   old_pixels=pa[keep][selected], new_pixels=pb[keep][selected])
   136	        return fit
```

## 从实际支持取距离列、关联并创建成对因子

`F/standalone/SideScanMappedRangeBank.py`，18–41、110–132；SHA-256：`906575f81b92605382b52de5a1d7963f5f2790847608cde21dce1770d11db460`。

```text
    18	def received_feature_ranges(query, pixels):
    19	    if 'source_phase_centre' in query:
    20	        from SideScanProjectedSubmap import projected_feature_observations, projected_pixel_indices
    21	        result = projected_feature_observations(query, pixels)
    22	        yy, xx, _ = projected_pixel_indices(query, pixels)
    23	        result['range_columns'] = query['source_range_column'][yy, xx]
    24	        return result
    25	    if 'received_phase_centre' not in query:
    26	        raise ValueError('Actual received phase/range provenance required, not averaged scan origins')
    27	    p = np.asarray(pixels, float).reshape(-1, 2)
    28	    if not np.isfinite(p).all(): raise ValueError('Finite received feature pixels required')
    29	    x = np.rint(p[:, 0]).astype(int)
    30	    y = np.rint((p[:, 1]+.5)/query['scale']-.5).astype(int)
    31	    h, w = query['received_phase_centre'].shape
    32	    if np.any((x < 0)|(x >= w)|(y < 0)|(y >= h)):
    33	        raise ValueError('Features outside received support')
    34	    phase = query['received_phase_centre'][y, x]
    35	    if np.any(phase < 0): raise ValueError('Features in holes cannot invent received ranges')
    36	    col = query['received_range_column'][y, x].astype(int)
    37	    bins = query['bins']
    38	    ranges = np.where(col < bins, bins-.5-col, col-bins+.5)*query['max_range_m']/bins
    39	    return dict(origins=query['received_phase_origins'][phase], ranges=ranges,
    40	        packet_ids=query['received_packet_ids'][phase], beam_rows=query['received_beam_rows'][phase],
    41	        timestamps_s=query['received_timestamps_s'][phase], range_columns=col)
   110	        added = 0
   111	        for old_node, pairs in groups.items():
   112	            if len(pairs) < self.minimum_features: continue
   113	            # Spread range support before capping compute, never select by
   114	            # hidden position error, seabed label, or a true-pose residual.
   115	            order = np.argsort([a['range'] for a, _ in pairs])
   116	            if len(order) > 48:
   117	                order = order[np.linspace(0, len(order)-1, 48).round().astype(int)]
   118	            pairs = [pairs[i] for i in order]
   119	            old, new = [a for a, _ in pairs], [b for _, b in pairs]
   120	            ra = np.array([a['range'] for a in old]); rb = np.array([a['range'] for a in new])
   121	            baseline = float(np.linalg.norm(graph.poses[node, :2]-graph.poses[old_node, :2]))
   122	            data = MappedRangeChange(
   123	                np.array([(a['xy']+b['xy'])/2 for a, b in pairs]),
   124	                np.array([a['origin'] for a in old]), np.array([a['origin'] for a in new]),
   125	                np.array([a['z'] for a in old]), np.array([a['z'] for a in new]), ra, rb,
   126	                conservative_range_change_sigma(ra, rb, baseline), maximum_range_m=query['max_range_m'])
   127	            graph.add_mapped_range_change(old_node, node, data, score=float(match.get('score', 0.)))
   128	            tokens = [list(a['token']) for pair in pairs for a in pair]
   129	            self.used_tokens.update(tuple(x) for x in tokens)
   130	            self.trace.append(dict(source=old_node, target=node, features=len(pairs),
   131	                timestamp_s=float(query['timestamp']), received_tokens=tokens,
   132	                known_landmark_height_used=False))
```

## 成对距离交给 GTSAM

`F/standalone/SideScanMappedRangeGraph.py`，35–61；SHA-256：`c577a125bc684195e4f7c4fc8a9eb3d25e8b6a8ecbec9344d886005feb9b5a2c`。

```text
    35	    def add_mapped_range_change(self, source, target, observations, *, score=0.):
    36	        if not isinstance(observations, MappedRangeChange):
    37	            raise TypeError('Only validated immutable received range observations allowed')
    38	        # Zero legacy SE2 precision is intentional: this is NOT a relative
    39	        # pose pseudo-measurement. The actual nonlinear CustomFactor below
    40	        # uses the raw ranges/XY and both variable endpoint poses.
    41	        super().add_edge(source, target, np.zeros(3), np.zeros((3, 3)),
    42	                         kind='mapped_range_change', score=score)
    43	        self.mapped_range_changes[id(self.edges[-1])] = observations
    44
    45	    def _edge_factor(self, edge):
    46	        if edge.kind != 'mapped_range_change':
    47	            return super()._edge_factor(edge)
    48	        data = self.mapped_range_changes[id(edge)]
    49	        source_key, target_key = self._key(edge.source), self._key(edge.target)
    50
    51	        def error_function(_factor, values, jacobians):
    52	            a = self._pose_array(values.atPose2(source_key))
    53	            b = self._pose_array(values.atPose2(target_key))
    54	            error, ja, jb = profiled_range_residual_jacobians(a, b, data)
    55	            if jacobians is not None:
    56	                jacobians[0] = np.asfortranarray(ja@self._local_delta_to_world(a))
    57	                jacobians[1] = np.asfortranarray(jb@self._local_delta_to_world(b))
    58	            return error
    59
    60	        return self._gtsam.CustomFactor(self._gtsam.noiseModel.Isotropic.Sigma(data.features, 1.),
    61	                                        [source_key, target_key], error_function)
```
