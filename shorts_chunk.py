# Dileep Explains: ONE Short's lip-sync split over many GitHub runners at once (23 Nov 2026 test; workflow shorts_chunked.yml).
#   python shorts_chunk.py plan  <name>            word timing once, face windows, the face/voice inputs of every window -> R2, matrix -> $GITHUB_OUTPUT
#   python shorts_chunk.py chunk <name> <n> <seg>  LatentSync 1.5 (12 steps + DeepCache, seed 1247) on CPU for ONE 16-frame segment of window n.
#                                                  Whisper features + face tracking run over the WHOLE window (same noise, same context as an
#                                                  unchunked run), only the denoising is limited to this segment -> the only seams are
#                                                  LatentSync's own 16-frame boundaries, exactly where a Kaggle run has them too.
#   python shorts_chunk.py join  <name>            segments -> fl<n>.mp4 per window (frame count checked) + seam check, then shorts_render.py assemble
# All media is read/written in R2 with boto3 + the repo secrets: nothing private in the repo, an artifact or the public log.
import os, sys, json, subprocess, time, random, re, math, glob
CMD, NAME = sys.argv[1], sys.argv[2]
PREFIX = os.environ.get("JOB_PREFIX", "gh_test"); assert PREFIX.startswith("gh_")
BUCKET = "eduorbex-media"
import boto3
from botocore.config import Config
S3 = boto3.client("s3", endpoint_url=os.environ["CF_R2_ENDPOINT"], aws_access_key_id=os.environ["CF_R2_ACCESS_KEY_ID"],
                  aws_secret_access_key=os.environ["CF_R2_SECRET_ACCESS_KEY"], region_name="auto", config=Config(signature_version="s3v4"))
JOB = json.loads(S3.get_object(Bucket=BUCKET, Key=f"s2work/shorts1/{PREFIX}/jobs/{NAME}.json")["Body"].read())
OUT = JOB["out"]; assert OUT.startswith(f"s2work/shorts1/{PREFIX}/")
FACE = JOB.get("face_from", OUT + "face/"); WORK = OUT + "work/"; CHUNKS = OUT + "chunks/"
ROOT = "/tmp/sc"; IN = f"{ROOT}/in"; T = f"{ROOT}/w"; LS = f"{ROOT}/ls"
for d in (IN, T): os.makedirs(d, exist_ok=True)
T0 = time.time(); FPS = 25; SEG = 16
def sh(c, quiet=False):   # commands hold local paths only
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    if not quiet or r.returncode: print(f"+ [{time.time() - T0:.0f}s]", c[:160], "-> rc", r.returncode, flush=True)
    if r.returncode: print(r.stdout[-800:], r.stderr[-2000:], flush=True)
    return r
def dur(p): return float(subprocess.run(f"ffprobe -v error -show_entries format=duration -of csv=p=0 {p}", shell=True, capture_output=True, text=True).stdout.strip() or 0)
def nframes(p): return int(subprocess.run(f"ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 {p}", shell=True, capture_output=True, text=True).stdout.strip() or 0)
def get(key, path):
    for t in range(4):
        try: S3.download_file(BUCKET, key, path); return
        except Exception as e: print("retry", os.path.basename(path), type(e).__name__, flush=True); time.sleep(5 * (t + 1))
    raise RuntimeError("download failed " + os.path.basename(path))
def put(path, key, ctype="application/octet-stream"): S3.upload_file(path, BUCKET, key, ExtraArgs={"ContentType": ctype})
def put_json(obj, key): S3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(obj).encode(), ContentType="application/json")
def mins(a, b=None): return round(((b or time.time()) - a) / 60, 2)

if CMD == "plan":
    if JOB.get("final"):   # production: refuse at once (before ~19 runners start) if this Short is already in out/ - never overwrite
        assert PREFIX == "gh_prod" and JOB["final"]["video"] == f"s2work/shorts1/out/sh_{NAME}.mp4", "final key must be this Short's own"
        for k in JOB["final"].values():
            try: S3.head_object(Bucket=BUCKET, Key=k); print("REFUSED: this Short is already in out/", flush=True); sys.exit(3)
            except S3.exceptions.ClientError: pass
    get(JOB["files"]["voice.flac"], f"{IN}/voice.flac"); get(JOB["files"]["face.mp4"], f"{IN}/face.mp4")
    sh(f"ffmpeg -y -loglevel error -i {IN}/voice.flac -af apad=pad_dur=0.2 -ar 48000 -ac 2 {T}/voice.wav"); D = dur(f"{T}/voice.wav")
    sh(f"ffmpeg -y -loglevel error -i {T}/voice.wav -ar 16000 -ac 1 {T}/v16.wav")
    t_al = time.time()
    import soundfile as sf, torch, uroman as ur
    from transformers import Wav2Vec2ForCTC, AutoTokenizer
    U = ur.Uroman(); R = "MahmoudAshraf/mms-300m-1130-forced-aligner"
    model = Wav2Vec2ForCTC.from_pretrained(R).eval(); tk = AutoTokenizer.from_pretrained(R); V = tk.get_vocab(); blank = V.get("<blank>", tk.pad_token_id)
    def norm(w): return "".join(c for c in w.lower() if (c in V and c.isalpha()) or (c == "'" and c in V))
    x = torch.from_numpy(sf.read(f"{T}/v16.wav", dtype="float32")[0]); SR = 16000; HOP = 320; WIN = 20 * SR; CTX = 2 * SR; ems = []; xn = (x - x.mean()) / (x.std() + 1e-7)
    for s in range(0, x.numel(), WIN):
        a = max(0, s - CTX); b = min(x.numel(), s + WIN + CTX)
        with torch.inference_mode(): e = torch.log_softmax(model(xn[a:b][None]).logits[0].float(), -1)
        r = e.size(0) / ((b - a) / HOP); f0 = int(round((s - a) / HOP * r)); f1 = int(round((min(s + WIN, x.numel()) - a) / HOP * r)); ems.append(e[f0:f1])
    em = torch.cat(ems); ratio = x.numel() / em.size(0) / SR
    raw, lineof = [], []
    for k, ln in enumerate(JOB["lines"]):
        for w in ln.split(): raw.append(w); lineof.append(k)
    rom = [norm(U.romanize_string(w)) for w in raw]; toks, owner = [], []
    for i, w in enumerate(rom):
        for c in w: toks.append(V[c]); owner.append(i)
    Tn, N = em.size(0), len(toks); tg = torch.tensor(toks)
    prev = torch.full((N + 1,), -1e9); prev[0] = 0; bp = torch.zeros((Tn, N + 1), dtype=torch.bool); eb = em[:, blank]
    for t in range(Tn):
        stay = prev + eb[t]; adv = torch.full_like(stay, -1e9); adv[1:] = prev[:-1] + em[t, tg]; bp[t] = adv > stay; prev = torch.maximum(stay, adv)
    j = N; fr = [0] * N
    for t in range(Tn - 1, -1, -1):
        if j > 0 and bp[t, j]: fr[j - 1] = t; j -= 1
    tm = {}
    for k, i in enumerate(owner): a0, a1 = tm.get(i, (fr[k], fr[k])); tm[i] = (min(a0, fr[k]), max(a1, fr[k]))
    words = []
    for i in range(len(raw)): words.append([tm[i][0] * ratio, (tm[i][1] + 1) * ratio] if i in tm else None)
    for i in range(len(words)):
        if words[i] is None:
            p = next((words[k][1] for k in range(i - 1, -1, -1) if words[k]), 0.0); q = next((words[k][0] for k in range(i + 1, len(words)) if words[k]), D)
            words[i] = [p, max(p + 0.2, q)]
    lines_t = []
    for k in range(len(JOB["lines"])):
        ws = [words[i] for i in range(len(raw)) if lineof[i] == k]; lines_t.append([ws[0][0], ws[-1][1]])
    FW = []
    for n, k in enumerate(JOB["face"]):
        a = max(0.0, lines_t[k][0] - 0.15); b = min(D, lines_t[k][1] + 0.25)
        if FW and a - FW[-1][2] < 0.6: FW[-1] = (FW[-1][0], FW[-1][1], b)
        else: FW.append((len(FW), a, b))
    print("aligned unmatched", j, "face windows", [(round(a, 2), round(b, 2)) for _, a, b in FW], flush=True)
    t_al = mins(t_al)
    cx, cy, cs = JOB["face_crop"]; FD = dur(f"{IN}/face.mp4") or 30
    inc = []; plan = {"windows": []}
    FWIN = JOB.get("face_win") or []   # owner 23 Nov (outfit rotation): eyes-open, no-smile windows (s) of the outfit clip
    def face_in(n, L):   # -> (start s, source) for face line n of length L; a line never runs past a window
        if not FWIN: return (n * FD / max(1, len(FW)) + 1.3 * n) % max(0.1, FD - L - 1), f"{IN}/face.mp4"   # old rule (no windows in the job)
        wa, wb = FWIN[n % len(FWIN)]; wl = max(0.2, wb - wa)
        if L + 0.5 <= wl: return wa + (1.3 * n) % max(0.01, wl - L - 0.5), f"{IN}/face.mp4"
        lp = f"{T}/fwin{n % len(FWIN)}.mp4"   # line longer than the window: loop INSIDE it (forward then reverse, no jump)
        if not os.path.exists(lp):
            sh(f"ffmpeg -y -loglevel error -ss {wa:.3f} -t {wl:.3f} -i {IN}/face.mp4 -filter_complex '[0:v]split[f][r];[r]reverse[rv];[f][rv]concat=n=2:v=1:a=0[o]' -map '[o]' -an -c:v libx264 -crf 12 -preset veryfast -pix_fmt yuv420p {lp}", quiet=True)
        return (1.3 * n) % wl, lp
    for n, a, b in FW:   # the same face/voice inputs the Kaggle kernel makes for window n
        L = b - a; fo, fsrc = face_in(n, L)
        sh(f"ffmpeg -y -loglevel error -ss {a:.3f} -t {L:.3f} -i {T}/v16.wav {T}/fa{n}.wav", quiet=True)
        sh(f"ffmpeg -y -loglevel error -stream_loop -1 -ss {fo:.2f} -i {fsrc} -t {L + 0.5:.3f} -vf 'crop={cs}:{cs}:{cx}:{cy},scale=512:512,fps=25' -an -c:v libx264 -crf 12 -preset veryfast -pix_fmt yuv420p {T}/fv{n}.mp4", quiet=True)
        put(f"{T}/fa{n}.wav", WORK + f"fa{n}.wav", "audio/wav"); put(f"{T}/fv{n}.mp4", WORK + f"fv{n}.mp4", "video/mp4")
        nseg = math.ceil((L * FPS + 3) / SEG)   # whisper gives ~L*25+1 frames; one spare segment at most (it exits at once if empty)
        plan["windows"].append({"n": n, "a": a, "b": b, "L": L, "segs": nseg}); inc += [{"n": n, "s": s} for s in range(nseg)]
    put_json({"words": words, "lines_t": lines_t, "FW": FW}, FACE + "align.json")
    plan["align_min"] = t_al; plan["plan_min"] = mins(T0); plan["chunks"] = len(inc); put_json(plan, WORK + "plan.json")
    print("chunks", len(inc), "plan min", plan["plan_min"], flush=True)
    with open(os.environ.get("GITHUB_OUTPUT", "/dev/null"), "a") as g: g.write("matrix=" + json.dumps({"include": inc}) + "\n")

elif CMD == "chunk":
    n, s = int(sys.argv[3]), int(sys.argv[4]); st = {"n": n, "seg": s}
    get(WORK + f"fa{n}.wav", f"{T}/fa{n}.wav"); get(WORK + f"fv{n}.mp4", f"{T}/fv{n}.mp4")
    t1 = time.time()
    LS_SHA = "a229c3948406bc2cf6eaf4873e662e70c6a04746"
    sh(f"git clone -q https://github.com/bytedance/LatentSync {LS} && cd {LS} && git -c advice.detachedHead=false checkout -q {LS_SHA}")
    def patch(f, a, b):
        p = f"{LS}/{f}"; x = open(p).read(); assert a in x, (f, a[:50]); open(p, "w").write(x.replace(a, b, 1))
    patch("scripts/inference.py", 'device="cuda",', 'device="cpu",')
    patch("scripts/inference.py", ').to("cuda")', ').to("cpu")')
    patch("latentsync/pipelines/lipsync_pipeline.py", 'ImageProcessor(height, device="cuda", mask_image=mask_image)', 'ImageProcessor(height, device="cpu", mask_image=mask_image)')
    # only THIS segment is denoised; restore uses the matching frames / boxes / affine matrices
    patch("latentsync/pipelines/lipsync_pipeline.py", 'for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):',
          '_s0 = int(os.environ.get("LS_SEG0", "0")); _s1 = min(num_inferences, int(os.environ.get("LS_SEG1", "99999")))\n'
          '        print("NUM_INFERENCES", num_inferences, "FRAMES", len(whisper_chunks), "PREP_DONE", __import__("time").time(), flush=True)\n'
          '        if _s0 >= _s1: open(video_out_path + ".empty", "w").write(str(num_inferences)); return\n'
          '        for i in tqdm.tqdm(range(_s0, _s1), desc="Doing inference..."):')
    patch("latentsync/pipelines/lipsync_pipeline.py", "synced_video_frames = self.restore_video(torch.cat(synced_video_frames), video_frames, boxes, affine_matrices)",
          'print("DENOISE_DONE", __import__("time").time(), flush=True); _o = _s0 * num_frames\n'
          "        synced_video_frames = self.restore_video(torch.cat(synced_video_frames), video_frames[_o:], boxes[_o:], affine_matrices[_o:])")
    p = f"{LS}/latentsync/utils/image_processor.py"; x = open(p).read()
    x2 = re.sub(r'if device == "cpu":\n\s+self\.face_detector = None\n\s+else:\n\s+self\.face_detector = FaceDetector\(device=device\)', "self.face_detector = FaceDetector(device=device)", x)
    assert x2 != x; open(p, "w").write(x2)
    patch("latentsync/utils/face_detector.py", 'providers=["CUDAExecutionProvider"]', 'providers=["CPUExecutionProvider"]')
    patch("latentsync/utils/face_detector.py", "ctx_id=cuda_to_int(device)", "ctx_id=-1")
    patch("latentsync/utils/affine_transform.py", 'device="cpu", dtype=torch.float16)', 'device="cpu", dtype=torch.float32)')
    os.chdir(LS); t2 = time.time()
    random.seed(n * 100 + s)
    for k in range(8):   # 20 runners pull the same weights at once: back off with jitter on errors (HF 429)
        sh("python -c \"from huggingface_hub import hf_hub_download as d; [d('ByteDance/LatentSync-1.5', f, local_dir='checkpoints') for f in ('whisper/tiny.pt', 'latentsync_unet.pt')]\"")
        if os.path.exists("checkpoints/latentsync_unet.pt"): break
        time.sleep(10 * (k + 1) + random.uniform(0, 15))
    sh("python -c \"from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', root='checkpoints/auxiliary', providers=['CPUExecutionProvider'])\"")
    st["code_min"] = mins(t1, t2); st["weights_min"] = mins(t2)
    o = f"{T}/fl{n}_{s:02d}.mp4"; t3 = time.time()
    r = subprocess.run(f"python -m scripts.inference --unet_config_path configs/unet/stage2.yaml --inference_ckpt_path checkpoints/latentsync_unet.pt --inference_steps {JOB.get('steps', 12)} --guidance_scale 1.5 --seed 1247 "
                       f"{'--enable_deepcache ' if JOB.get('deepcache', True) else ''}--video_path {T}/fv{n}.mp4 --audio_path {T}/fa{n}.wav --video_out_path {o}",
                       shell=True, capture_output=True, text=True, env=dict(os.environ, LS_SEG0=str(s), LS_SEG1=str(s + 1)))
    lg = r.stdout + r.stderr; t4 = time.time()
    mm = re.search(r"NUM_INFERENCES (\d+) FRAMES (\d+) PREP_DONE ([0-9.]+)", lg); md = re.search(r"DENOISE_DONE ([0-9.]+)", lg)
    if mm: st.update(num_inferences=int(mm.group(1)), frames=int(mm.group(2)), prep_min=mins(t3, float(mm.group(3))))
    if mm and md: st["denoise_min"] = mins(float(mm.group(3)), float(md.group(1))); st["restore_write_min"] = mins(float(md.group(1)), t4)
    st["inference_total_min"] = mins(t3, t4); st["rc"] = r.returncode
    if os.path.exists(o + ".empty"): st["empty"] = True; print("segment", s, "is past the end of window", n, "-> nothing to do", flush=True)
    elif os.path.exists(o):
        st["out_frames"] = nframes(o); put(o, CHUNKS + os.path.basename(o), "video/mp4")
    else:
        print(lg[-3000:], flush=True)
    st["chunk_min"] = mins(T0); put_json(st, CHUNKS + f"fl{n}_{s:02d}.json"); print("CHUNK", json.dumps(st), flush=True)
    if r.returncode and not st.get("empty"): sys.exit(1)

elif CMD == "join":
    import numpy as np
    plan = json.loads(S3.get_object(Bucket=BUCKET, Key=WORK + "plan.json")["Body"].read())
    rep = {"windows": []}
    for w in plan["windows"]:
        n = w["n"]; parts = []; info = []
        for s in range(w["segs"]):
            try: info.append(json.loads(S3.get_object(Bucket=BUCKET, Key=CHUNKS + f"fl{n}_{s:02d}.json")["Body"].read()))
            except Exception: info.append({"n": n, "seg": s, "missing": True}); continue
            if info[-1].get("empty"): continue
            p = f"{T}/fl{n}_{s:02d}.mp4"
            try: get(CHUNKS + f"fl{n}_{s:02d}.mp4", p); parts.append(p)
            except Exception: info[-1]["missing"] = True
        want = next((i["frames"] for i in info if i.get("frames")), None)
        exp_segs = math.ceil(want / SEG) if want else None
        complete = exp_segs is not None and len(parts) == exp_segs and all(not i.get("missing") for i in info[:exp_segs])
        open(f"{T}/l{n}.txt", "w").write("\n".join(f"file '{p}'" for p in parts))
        fl = f"{T}/fl{n}.mp4"
        if complete:
            sh(f"ffmpeg -y -loglevel error -f concat -safe 0 -i {T}/l{n}.txt -an -c:v libx264 -crf 14 -preset veryfast -pix_fmt yuv420p {fl}")
        got = nframes(fl) if os.path.exists(fl) else 0
        # seam check: mouth-area mean abs difference between consecutive frames, at segment boundaries vs inside segments
        seam = {}
        if got:
            raw = subprocess.run(f"ffmpeg -loglevel error -i {fl} -vf 'crop=204:144:154:240,format=gray' -f rawvideo -", shell=True, capture_output=True).stdout
            fr = np.frombuffer(raw, np.uint8).reshape(-1, 144, 204).astype(np.float32)
            d = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))   # d[i] = change from frame i to i+1
            b = [k * SEG - 1 for k in range(1, math.ceil(len(fr) / SEG))]
            inner = [d[i] for i in range(len(d)) if i not in b]
            med = float(np.median(inner)) if inner else 0
            seam = {"boundary_diffs": [round(float(d[i]), 2) for i in b if i < len(d)], "inside_median": round(med, 2), "inside_p95": round(float(np.percentile(inner, 95)), 2) if inner else 0,
                    "worst_boundary_ratio": round(max([float(d[i]) for i in b if i < len(d)] or [0]) / (med or 1), 2)}
            put(fl, FACE + f"fl{n}.mp4", "video/mp4")
        rep["windows"].append({"n": n, "frames_expected": want, "frames_joined": got, "complete": complete, "chunks": info, "seams": seam})
        print("window", n, "frames", got, "of", want, "complete", complete, "seams", seam, flush=True)
    put_json(rep, OUT + "join.json")
