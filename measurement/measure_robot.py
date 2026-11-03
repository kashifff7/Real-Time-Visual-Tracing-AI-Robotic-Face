import argparse, csv, glob, json, os, subprocess, sys, threading, time
from pathlib import Path

import numpy as np

OUT = Path("measurements")
FRAMES = OUT / "frames"
YUNET_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")

def now():
    return time.perf_counter()

def cpu_temp():
    try:
        out = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True).stdout
        return float(out.split("=")[1].split("'")[0])
    except Exception:
        try:
            return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000.0
        except Exception:
            return float("nan")

def throttled():
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout
        return out.strip().split("=")[1]
    except Exception:
        return "n/a"

def open_camera(w=320, h=240):
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_preview_configuration(main={"format": "RGB888", "size": (w, h)}))
        cam.start()
        time.sleep(1.0)
        return lambda: cam.capture_array(), cam.stop
    except Exception:
        import cv2
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if not cap.isOpened():
            sys.exit("No camera found (picamera2 and cv2.VideoCapture(0) both failed).")

        def grab():
            ok, f = cap.read()
            if not ok:
                raise RuntimeError("camera read failed")
            return f
        return grab, cap.release

class Servos:

    def __init__(self, enabled):
        self.enabled = enabled
        if enabled:
            from adafruit_servokit import ServoKit
            self.kit = ServoKit(channels=16)

    def write(self, hor, ver):
        if self.enabled:
            self.kit.servo[0].angle = hor
            self.kit.servo[1].angle = ver

def map_range(x, in_min, in_max, out_min, out_max):
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

HAAR_URL = ("https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
            "haarcascade_frontalface_default.xml")

def haar():
    import cv2, urllib.request
    p = os.path.join(getattr(getattr(cv2, "data", None), "haarcascades", ""), "haarcascade_frontalface_default.xml")
    if not os.path.exists(p):
        p = str(OUT / "haarcascade_frontalface_default.xml")
        if not os.path.exists(p):
            OUT.mkdir(exist_ok=True)
            urllib.request.urlretrieve(HAAR_URL, p)
    c = cv2.CascadeClassifier(p)
    if c.empty():
        sys.exit("could not load Haar cascade " + p)
    return c

def yunet(w, h):
    import cv2, urllib.request
    p = OUT / "face_detection_yunet_2023mar.onnx"
    if not p.exists():
        OUT.mkdir(exist_ok=True)
        print("downloading YuNet model ...")
        urllib.request.urlretrieve(YUNET_URL, p)
    return cv2.FaceDetectorYN.create(str(p), "", (w, h), 0.6, 0.3, 5000)

def largest(boxes):
    return max(boxes, key=lambda b: b[2] * b[3]) if len(boxes) else None

def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    u = aw * ah + bw * bh - inter
    return inter / u if u else 0.0

def cmd_track(a):
    import cv2, psutil
    grab, close = open_camera(a.width, a.height)
    det = haar()
    servo = Servos(not a.no_servo)
    fdir = FRAMES / a.condition
    fdir.mkdir(parents=True, exist_ok=True)
    log = OUT / f"track_{a.condition}.csv"
    psutil.cpu_percent(None)
    W, H = a.width, a.height
    hor, ver = 90, 90
    servo.write(hor, ver)
    time.sleep(0.5)
    rows = []
    print(f"tracking {a.frames} frames, condition '{a.condition}' (robot's eye_module logic) ...")
    for i in range(a.frames):
        t0 = now()
        frame = grab()
        t1 = now()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        t2 = now()
        faces = det.detectMultiScale(gray, 1.3, 5)
        t3 = now()
        cx = cy = float("nan")
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cx, cy = x + w // 2, y + h // 2
            hor = map_range(cx, 0, W, 60, 120)
            ver = map_range(cy, 0, H, 120, 60)
        t4 = now()
        if len(faces) > 0:
            servo.write(hor, ver)
        t5 = now()
        if a.display:
            cv2.imshow("Eye Tracking", frame); cv2.waitKey(1)
        t6 = now()
        if a.save_every and i % a.save_every == 0:
            cv2.imwrite(str(fdir / f"{i:06d}.jpg"), frame)
        rows.append(dict(frame=i, t=t0, cap=1e3 * (t1 - t0), gray=1e3 * (t2 - t1),
                         det=1e3 * (t3 - t2), map=1e3 * (t4 - t3), servo=1e3 * (t5 - t4),
                         display=1e3 * (t6 - t5), total=1e3 * (t6 - t0), face=int(len(faces) > 0),
                         raw_x=cx, raw_y=cy, err_x=cx - W / 2, err_y=cy - H / 2, ang_x=hor, ang_y=ver,
                         cpu=psutil.cpu_percent(None), temp=cpu_temp() if i % 30 == 0 else ""))
    close()
    with open(log, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)
    tot = np.array([r["total"] for r in rows])
    print(f"saved {log}: median T_track {np.median(tot):.1f} ms, "
          f"mean {tot.mean():.1f} ms -> {1000 / tot.mean():.1f} FPS")

def cmd_annotate(a):
    import cv2
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    conds = [a.condition] if a.condition else sorted(p.name for p in FRAMES.iterdir() if p.is_dir())
    Z = 2
    st = {"p1": None, "p2": None, "drag": False}

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            st.update(p1=(x, y), p2=(x, y), drag=True)
        elif ev == cv2.EVENT_MOUSEMOVE and st["drag"]:
            st["p2"] = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP and st["drag"]:
            st.update(p2=(x, y), drag=False)

    cv2.namedWindow("annotate", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("annotate", on_mouse)
    stop = False
    for c in conds:
        if stop:
            break
        gt = FRAMES / c / "ground_truth.csv"
        done = {}
        if gt.exists():
            done = {r["file"]: r for r in csv.DictReader(open(gt))}
        files = sorted(p.name for p in (FRAMES / c).glob("*.jpg"))
        todo = [f for f in files if f not in done]
        print(f"{c}: {len(todo)} frames to annotate")
        for n_, fn in enumerate(todo):
            img = cv2.resize(cv2.imread(str(FRAMES / c / fn)), None, fx=Z, fy=Z)
            st.update(p1=None, p2=None, drag=False)
            result = None
            while result is None:
                view = img.copy()
                if st["p1"] and st["p2"]:
                    cv2.rectangle(view, st["p1"], st["p2"], (0, 255, 0), 2)
                cv2.putText(view, f"{c}  {n_ + 1}/{len(todo)}   drag box | ENTER save | n no face | r redo | ESC stop",
                            (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.imshow("annotate", view)
                k = cv2.waitKey(20) & 0xFF
                if k == 27:
                    result = "stop"
                elif k == ord("n"):
                    result = dict(file=fn, face=0, x="", y="", w="", h="")
                elif k == ord("r"):
                    st.update(p1=None, p2=None, drag=False)
                elif k in (13, 10, 32) and st["p1"] and st["p2"]:
                    (x1, y1), (x2, y2) = st["p1"], st["p2"]
                    x, y = min(x1, x2) / Z, min(y1, y2) / Z
                    w, h = abs(x2 - x1) / Z, abs(y2 - y1) / Z
                    if w > 3 and h > 3:
                        result = dict(file=fn, face=1, x=x, y=y, w=w, h=h)
            if result == "stop":
                stop = True
                break
            done[fn] = result
            with open(gt, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=["file", "face", "x", "y", "w", "h"])
                wr.writeheader()
                wr.writerows(done.values())
    cv2.destroyAllWindows()
    print("annotation saved (run again to continue where you stopped)")

def cmd_detect(a):
    import cv2, psutil
    hdet = haar()
    ydet = None
    rows = []
    for gt in sorted(FRAMES.glob("*/ground_truth.csv")):
        cond = gt.parent.name
        truth = [r for r in csv.DictReader(open(gt))]
        imgs = [cv2.imread(str(gt.parent / r["file"])) for r in truth]
        h, w = imgs[0].shape[:2]
        if ydet is None:
            ydet = yunet(w, h)
        for name in ("haar", "yunet"):
            hits = n_face = 0
            psutil.cpu_percent(None)
            t0 = now()
            for r, img in zip(truth, imgs):
                if name == "haar":
                    boxes = hdet.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                                  scaleFactor=a.scale, minNeighbors=a.neighbors)
                    b = largest(boxes)
                else:
                    _, faces = ydet.detect(img)
                    b = None if faces is None else largest([f[:4] for f in faces])
                if int(r["face"]):
                    n_face += 1
                    ref = tuple(float(r[k]) for k in "xywh")
                    if b is not None and iou(tuple(map(float, b)), ref) >= 0.5:
                        hits += 1
            dt = now() - t0
            rows.append(dict(condition=cond, detector=name, frames=len(imgs), face_frames=n_face,
                             accuracy=100.0 * hits / n_face if n_face else float("nan"),
                             fps=len(imgs) / dt, cpu=psutil.cpu_percent(None)))
            print(rows[-1])
    if not rows:
        sys.exit("no annotated frames found - run track and annotate first")
    with open(OUT / "detect.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)

def cmd_conv(a):
    import requests, pyttsx3
    key = os.environ.get("OPENAI_API_KEY") or sys.exit("set OPENAI_API_KEY")
    hdr = {"Authorization": f"Bearer {key}"}
    wavs = sorted(glob.glob(os.path.join(a.questions, "*.wav")))
    if not wavs:
        sys.exit("put the fixed question recordings (5 s, 16 kHz mono WAV, like the robot's) in " + a.questions)
    stress = None
    if a.load:
        n = max(1, round(4 * a.load / 100))
        stress = subprocess.Popen(["stress-ng", "--cpu", str(n), "--cpu-load", str(a.load)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
    engine = pyttsx3.init()
    engine.setProperty("rate", 120)
    engine.setProperty("volume", 0.9)
    for v in engine.getProperty("voices"):
        if "english" in (v.name or "").lower() or "en" == (v.id or "")[:2] or "/en" in (v.id or ""):
            engine.setProperty("voice", v.id)
            break
    started = {}
    engine.connect("started-utterance", lambda name: started.setdefault("t", now()))
    log = OUT / "conv.csv"
    new = not log.exists()
    OUT.mkdir(exist_ok=True)
    with open(log, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["load", "question", "repeat", "t_stt", "t_llm", "t_tts", "t_conv", "text", "reply_chars"])
        for rep in range(a.repeats):
            for wv in wavs:
                try:
                    t0 = now()
                    try:
                        r = requests.post("https://api.openai.com/v1/audio/transcriptions", headers=hdr,
                                          files={"file": (os.path.basename(wv), open(wv, "rb")),
                                                 "model": (None, "whisper-1"),
                                                 "language": (None, "en")}, timeout=60)
                        js = r.json()
                        if "text" not in js:
                            if stress: stress.terminate()
                            sys.exit("Whisper API error: " + json.dumps(js.get("error", js)))
                        text = js["text"]
                        t1 = now()
                        r = requests.post("https://api.openai.com/v1/chat/completions", headers=hdr, timeout=60,
                                          json={"model": a.model, "temperature": 0.7, "max_tokens": 100, "messages": [
                                              {"role": "system", "content": "You are a precise voice assistant. Respond concisely in 1-2 short sentences. Always reply in English."},
                                              {"role": "user", "content": text}]})
                        js = r.json()
                        if "choices" not in js:
                            if stress: stress.terminate()
                            sys.exit("Chat API error: " + json.dumps(js.get("error", js)))
                        reply = js["choices"][0]["message"]["content"]
                        t2 = now()
                    except requests.exceptions.RequestException as e:
                        wr.writerow([a.load, os.path.basename(wv), rep, "nan", "nan", "nan", "nan",
                                     "FAILED: " + type(e).__name__, 0])
                        f.flush()
                        print(f"load {a.load}% {os.path.basename(wv)}: FAILED ({type(e).__name__}) - recorded, continuing")
                        time.sleep(3)
                        continue
                    started.clear()
                    engine.say(reply)
                    engine.runAndWait()
                    t3 = started.get("t", float("nan"))
                    row = [a.load, os.path.basename(wv), rep, 1e3 * (t1 - t0), 1e3 * (t2 - t1),
                           1e3 * (t3 - t2), 1e3 * (t3 - t0), text, len(reply)]
                    wr.writerow(row)
                    f.flush()
                    print(f"load {a.load}% {os.path.basename(wv)}: STT {row[3]:.0f}  LLM {row[4]:.0f}  "
                          f"TTS {row[5]:.0f}  total {row[6]:.0f} ms")
                    print(f"    heard: {text}")
                    print(f"    reply: {reply}")
                except Exception as e:
                    print(f"load {a.load}% {os.path.basename(wv)}: skipped ({type(e).__name__}) - network problem, continuing")
                    wr.writerow([a.load, os.path.basename(wv), rep, "", "", "", "", "ERROR " + type(e).__name__, ""])
                    f.flush()
    if stress:
        stress.terminate()

def cmd_thermal(a):
    import psutil
    OUT.mkdir(exist_ok=True)
    log = OUT / f"thermal_run{a.run}.csv"
    end = time.time() + a.hours * 3600
    t_start = time.time()
    with open(log, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["minute", "temp_c", "throttled", "cpu_percent"])
        while time.time() < end:
            wr.writerow([round((time.time() - t_start) / 60, 2), cpu_temp(), throttled(),
                         psutil.cpu_percent(interval=1)])
            f.flush()
            time.sleep(max(0, a.interval - 1))
    print("saved", log, "- record failure events by hand in measurements/failures.csv "
          "(run,minute,type,notes)")

def _ask_once(wv, hdr, model, engine, started):
    import requests
    t0 = now()
    r = requests.post("https://api.openai.com/v1/audio/transcriptions", headers=hdr,
                      files={"file": (os.path.basename(wv), open(wv, "rb")),
                             "model": (None, "whisper-1"), "language": (None, "en")}, timeout=60)
    js = r.json()
    if "text" not in js:
        return None, None, None, "whisper: " + json.dumps(js.get("error", js))[:150]
    t1 = now()
    r = requests.post("https://api.openai.com/v1/chat/completions", headers=hdr, timeout=60,
                      json={"model": model, "temperature": 0.7, "max_tokens": 100, "messages": [
                          {"role": "system", "content": "You are a precise voice assistant. Respond concisely "
                                                        "in 1-2 short sentences. Always reply in English."},
                          {"role": "user", "content": js["text"]}]})
    js = r.json()
    if "choices" not in js:
        return None, None, None, "chat: " + json.dumps(js.get("error", js))[:150]
    t2 = now()
    started.clear()
    engine.say(js["choices"][0]["message"]["content"])
    engine.runAndWait()
    t3 = started.get("t", float("nan"))
    return 1e3 * (t1 - t0), 1e3 * (t2 - t1), 1e3 * (t3 - t2), None

def cmd_overnight(a):
    import cv2, psutil, threading, io, contextlib, shutil
    OUT.mkdir(exist_ok=True)
    run = a.run
    t_start, t_end = time.time(), time.time() + a.hours * 3600
    minute = lambda: round((time.time() - t_start) / 60, 2)
    stop = threading.Event()
    lock = threading.Lock()

    fail_path = OUT / "failures.csv"
    if not fail_path.exists():
        with open(fail_path, "w", newline="") as f:
            csv.writer(f).writerow(["run", "minute", "type", "notes"])

    def failure(kind, notes):
        with lock:
            with open(fail_path, "a", newline="") as f:
                csv.writer(f).writerow([run, minute(), kind, notes])
        print(f"[FAILURE] minute {minute()}: {kind} - {notes}")

    def thermal():
        with open(OUT / f"thermal_run{run}.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["minute", "temp_c", "throttled", "cpu_percent"])
            while not stop.is_set():
                wr.writerow([minute(), cpu_temp(), throttled(), psutil.cpu_percent(interval=1)])
                f.flush()
                stop.wait(max(0, a.interval - 1))

    def conversation():
        key = os.environ.get("OPENAI_API_KEY")
        wavs = sorted(glob.glob(os.path.join(a.questions, "*.wav")))
        if not key or not wavs:
            print("[INFO] no OPENAI_API_KEY or no questions/*.wav: running without conversation")
            return
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 120)
        engine.setProperty("volume", 0.9)
        for v in engine.getProperty("voices"):
            if "english" in (v.name or "").lower() or (v.id or "")[:2] == "en" or "/en" in (v.id or ""):
                engine.setProperty("voice", v.id)
                break
        started = {}
        engine.connect("started-utterance", lambda name: started.setdefault("t", now()))
        hdr = {"Authorization": f"Bearer {key}"}
        k = 0
        with open(OUT / f"conv_overnight_run{run}.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["minute", "question", "t_stt", "t_llm", "t_tts", "t_conv", "error"])
            while not stop.wait(a.ask_every * 60):
                wv = wavs[k % len(wavs)]; k += 1
                try:
                    t0 = now()
                    s_, l_, tts, err = _ask_once(wv, hdr, a.model, engine, started)
                    total = 1e3 * (now() - t0)
                    if err:
                        failure("conversation", err)
                    elif total > 15000:
                        failure("conversation", f"turn took {total / 1000:.1f} s (> 15 s)")
                    wr.writerow([minute(), os.path.basename(wv), s_, l_, tts,
                                 None if err else (s_ + l_ + tts), err or ""])
                    f.flush()
                except Exception as e:
                    failure("conversation", f"{type(e).__name__}: {e}"[:150])

    threads = [threading.Thread(target=thermal, daemon=True), threading.Thread(target=conversation, daemon=True)]
    for t in threads:
        t.start()

    det = haar()
    try:
        servo = Servos(not a.no_servo)
    except Exception as e:
        failure("servo/I2C", f"ServoKit init failed: {e}"[:150])
        servo = Servos(False)
    grab = close = None

    def open_cam():
        nonlocal grab, close
        try:
            if close:
                close()
        except Exception:
            pass
        grab, close = open_camera(320, 240)

    open_cam()
    log = open(OUT / f"overnight_run{run}.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["minute", "frames", "fps", "mean_ttrack_ms", "max_ttrack_ms", "face_frames", "cpu", "temp_c"])
    win_t, win_n, win_sum, win_max, win_face, frames = time.time(), 0, 0.0, 0.0, 0, 0
    last = time.time()
    print(f"overnight run {run}: {a.hours} h, logging to {OUT}/  (Ctrl+C stops early and still writes the report)")
    try:
        while time.time() < t_end:
            t0 = now()
            try:
                frame = grab()
            except Exception as e:
                failure("camera", f"{type(e).__name__}: {e}"[:150])
                time.sleep(2)
                try:
                    open_cam()
                except Exception as e2:
                    failure("camera", f"reopen failed: {e2}"[:150])
                    time.sleep(10)
                continue
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = det.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                cx, cy = x + w // 2, y + h // 2
                try:
                    servo.write(map_range(cx, 0, 320, 60, 120), map_range(cy, 0, 240, 120, 60))
                except Exception as e:
                    failure("servo/I2C", f"{type(e).__name__}: {e}"[:150])
                win_face += 1
            dt = 1e3 * (now() - t0)
            gap = time.time() - last
            if gap > 2.0:
                failure("freeze", f"no frame processed for {gap:.1f} s")
            last = time.time()
            frames += 1; win_n += 1; win_sum += dt; win_max = max(win_max, dt)
            if time.time() - win_t >= 60:
                wr.writerow([minute(), frames, round(win_n / (time.time() - win_t), 2), round(win_sum / win_n, 2),
                             round(win_max, 2), win_face, psutil.cpu_percent(None), cpu_temp()])
                log.flush()
                print(f"  minute {minute():7.1f}  {win_n / (time.time() - win_t):5.1f} FPS  "
                      f"temp {cpu_temp():.1f} C  throttled {throttled()}")
                win_t, win_n, win_sum, win_max, win_face = time.time(), 0, 0.0, 0.0, 0
    except KeyboardInterrupt:
        print("stopped early by user")
    finally:
        stop.set()
        log.close()
        try:
            close()
        except Exception:
            pass
        time.sleep(2)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            cmd_report(a)
        except Exception as e:
            print("report failed:", e)
    (OUT / "report.txt").write_text(buf.getvalue())
    print(buf.getvalue())
    dest = "/boot/firmware/results"
    files = [str(p) for p in OUT.glob("*.csv")] + [str(OUT / "report.txt")] + \
            [str(p) for p in (OUT / "figures").glob("*.png")]
    rc = subprocess.run(["sudo", "-n", "mkdir", "-p", dest]).returncode
    if rc == 0:
        subprocess.run(["sudo", "-n", "cp"] + files + [dest])
        print(f"results copied to {dest} (shows as D:\\results on Windows)")
    else:
        print(f"could not copy to {dest}; results are in {OUT}/")

def ms(x):
    return f"{np.mean(x):.1f}", f"{np.std(x, ddof=1):.1f}", f"{np.median(x):.1f}", f"{np.percentile(x, 95):.1f}"

def cmd_report(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    figs = OUT / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    tr = {}
    for p in sorted(OUT.glob("track_*.csv")):
        tr[p.stem[6:]] = list(csv.DictReader(open(p)))
    if tr:
        allrows = [r for v in tr.values() for r in v]
        print("\n=== Table 9.3  Per-stage visual tracking latency (%d frames) ===" % len(allrows))
        print(f"{'Stage':28s} {'Mean':>7s} {'SD':>7s} {'Median':>7s} {'P95':>7s}  (ms)")
        stages = [("Capture", "cap"), ("Mirror + grayscale", "gray"), ("Haar detection", "det"),
                  ("Angle mapping", "map"), ("Servo write (I2C)", "servo")]
        if np.mean([float(r.get("display", 0) or 0) for r in allrows]) > 1.0:
            stages.append(("Display", "display"))
        stages.append(("Total T_track", "total"))
        means = {}
        for lab, k in stages:
            x = np.array([float(r[k]) for r in allrows])
            means[k] = (x.mean(), x.std(ddof=1))
            print(f"{lab:28s} " + " ".join(f"{v:>7s}" for v in ms(x)))
        tot = np.array([float(r["total"]) for r in allrows])
        print(f"{'Resulting frame rate (FPS)':28s} {1000 / tot.mean():7.1f}")

        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.hist(tot, bins=40, color="0.4")
        ax.axvline(np.median(tot), ls="--", color="k")
        ax.set_xlabel("Visual tracking latency T$_{track}$ (ms)")
        ax.set_ylabel("Number of frames")
        fig.tight_layout(); fig.savefig(figs / "fig_latency_hist.png", dpi=300)

        fig, ax = plt.subplots(figsize=(4.5, 3))
        labs = [s[0] for s in stages[:-1]]
        ax.bar(labs, [means[s[1]][0] for s in stages[:-1]], yerr=[means[s[1]][1] for s in stages[:-1]],
               color="0.5", capsize=3)
        ax.set_ylabel("Time per frame (ms)")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        fig.tight_layout(); fig.savefig(figs / "fig_stage_breakdown.png", dpi=300)

        ex, ey = [], []
        for cond, rows in tr.items():
            gt = FRAMES / cond / "ground_truth.csv"
            if not gt.exists() or "step" in cond:
                continue
            first = next(iter(sorted((FRAMES / cond).glob("*.jpg"))), None)
            if first is None:
                continue
            import cv2
            H, W = cv2.imread(str(first)).shape[:2]
            for r in csv.DictReader(open(gt)):
                if int(r["face"]):
                    ex.append(float(r["x"]) + float(r["w"]) / 2 - W / 2)
                    ey.append(float(r["y"]) + float(r["h"]) / 2 - H / 2)
        if ex:
            ex, ey = np.abs(ex), np.abs(ey)
            ee = np.hypot(ex, ey)
            print(f"\n=== Table 9.4  Tracking error over {len(ee)} annotated frames (pixels from image centre) ===")
            print(f"{'':22s} {'horiz':>8s} {'vert':>8s} {'eucl':>8s}")
            for lab, f in (("Mean absolute error", np.mean), ("RMSE", lambda v: np.sqrt(np.mean(v ** 2))),
                           ("Maximum", np.max), ("95th percentile", lambda v: np.percentile(v, 95))):
                print(f"{lab:22s} {f(ex):8.2f} {f(ey):8.2f} {f(ee):8.2f}")
        for cond, rows in tr.items():
            if "step" not in cond:
                continue
            t = np.array([float(r["t"]) for r in rows]); t -= t[0]
            e = np.array([float(r["err_x"]) if r["err_x"] not in ("", "nan") else np.nan for r in rows])
            ang = np.array([float(r["ang_x"]) for r in rows])
            big = np.where(np.abs(e) > 30)[0]
            if not len(big):
                print(f"{cond}: no step (>30 px) found"); continue
            k0 = big[0]
            a0, a1 = ang[max(k0 - 1, 0)], np.median(ang[-max(5, len(ang) // 5):])
            d = a1 - a0
            def cross(p):
                idx = np.where((ang[k0:] - a0) / d >= p)[0] if d else []
                return t[k0 + idx[0]] if len(idx) else np.nan
            rise = cross(0.9) - cross(0.1)
            ok = np.abs(np.nan_to_num(e[k0:], nan=999)) < 10
            settle = np.nan
            for j in range(len(ok) - 10):
                if ok[j:j + 10].all():
                    settle = t[k0 + j] - t[k0]; break
            print(f"\nStep response '{cond}': eye moved {d:.1f} deg, rise time (10-90 %) {rise:.2f} s, "
                  f"settling time (|e| < 10 px) {settle:.2f} s")
        runs = [(c, r) for c, r in tr.items() if "step" not in c and "longrun" not in c]
        walk = [x for x in runs if "walk" in x[0]] or runs
        if walk:
            cond, rows = max(walk, key=lambda x: len(x[1]))
            t = np.array([float(r["t"]) for r in rows]); t -= t[0]
            e = [float(r["err_x"]) if r["err_x"] not in ("", "nan") else np.nan for r in rows]
            fig, ax = plt.subplots(figsize=(5.5, 3))
            ax.plot(t, e, "k-", lw=0.8, label="Tracking error")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Horizontal error (px)")
            ax2 = ax.twinx()
            ax2.plot(t, [float(r["ang_x"]) for r in rows], "k--", lw=0.8, label="Eye angle")
            ax2.set_ylabel("Eye angle (deg)")
            ax.legend(loc="upper left", frameon=False); ax2.legend(loc="upper right", frameon=False)
            ax.set_title(cond, fontsize=9)
            fig.tight_layout(); fig.savefig(figs / "fig_tracking_timeseries.png", dpi=300)

    dp = OUT / "detect.csv"
    if dp.exists():
        d = list(csv.DictReader(open(dp)))
        print("\n=== Table 9.1  Detection accuracy (%) per condition (Haar) ===")
        for r in d:
            if r["detector"] == "haar":
                print(f"{r['condition']:24s} {float(r['accuracy']):6.1f}  ({r['face_frames']} face frames)")
        acc = np.array([float(r["accuracy"]) for r in d if r["detector"] == "haar"])
        acc = acc[~np.isnan(acc)] if len(acc) else acc
        print(f"mean over conditions: {acc.mean():.1f} %  (SD {acc.std(ddof=1) if len(acc) > 1 else 0:.1f}),"
              f" best {acc.max():.1f} %, worst {acc.min():.1f} %")
        print("\n=== Table 9.6  Haar cascade vs YuNet (mean over conditions) ===")
        summ = {}
        for det in ("haar", "yunet"):
            rr = [r for r in d if r["detector"] == det]
            summ[det] = [np.mean([float(r[k]) for r in rr]) for k in ("accuracy", "fps", "cpu")]
            print(f"{det:6s} accuracy {summ[det][0]:.1f} %  FPS {summ[det][1]:.1f}  CPU {summ[det][2]:.0f} %")
        fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.6))
        for ax, i, lab in zip(axs, range(3), ("Accuracy (%)", "Frame rate (FPS)", "CPU load (%)")):
            ax.bar(["Haar", "YuNet"], [summ["haar"][i], summ["yunet"][i]], color=["0.6", "0.3"])
            ax.set_ylabel(lab)
        fig.tight_layout(); fig.savefig(figs / "fig_detector_compare.png", dpi=300)
        fig, ax = plt.subplots(figsize=(4.5, 3))
        lights = sorted({r["condition"].split("_")[0] for r in d})
        for L, mk in zip(lights, "osd^v"):
            def dist(c):
                try:
                    return float(c.split("_")[1].rstrip("m"))
                except (IndexError, ValueError):
                    return None
            pts = sorted((dist(r["condition"]), float(r["accuracy"]))
                         for r in d if r["detector"] == "haar" and r["condition"].startswith(L + "_")
                         and dist(r["condition"]) is not None)
            if pts:
                ax.plot(*zip(*pts), marker=mk, color="k", label=L)
        ax.set_xlabel("Distance (m)"); ax.set_ylabel("Detection accuracy (%)")
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(figs / "fig_accuracy.png", dpi=300)

    cp = OUT / "conv.csv"
    if cp.exists():
        c = list(csv.DictReader(open(cp)))
        print("\n=== Table 9.2  Conversational latency vs CPU load (mean ms) ===")
        print(f"{'load':>5s} {'n':>4s} {'t_STT':>8s} {'t_LLM':>8s} {'t_TTS':>8s} {'T_conv':>8s} {'median':>8s}")
        for L in sorted({int(r["load"]) for r in c}):
            rr = [r for r in c if int(r["load"]) == L]
            ok = lambda r: r["t_conv"] not in ("", "nan")
            bad = sum(1 for r in rr if not ok(r))
            rr = [r for r in rr if ok(r)]
            if not rr:
                continue
            m = [np.nanmean([float(r[k]) for r in rr]) for k in ("t_stt", "t_llm", "t_tts", "t_conv")]
            md = np.nanmedian([float(r["t_conv"]) for r in rr])
            if bad:
                print(f"      ({bad} turn(s) at {L}% failed with network errors and are excluded)")
            nf = sum(1 for r in rr if str(r["text"]).startswith("FAILED"))
            print(f"{L:5d} {len(rr):4d} " + " ".join(f"{v:8.0f}" for v in m) + f" {md:8.0f}"
                  + (f"   ({nf} failed/timed out)" if nf else ""))

    th = sorted(OUT.glob("thermal_run*.csv"))
    if th:
        print("\n=== Table 9.5  CPU temperature (run 1) and Table 13.2 max temperature per run ===")
        r1 = list(csv.DictReader(open(th[0])))
        for target in (0, 15, 30, 60, 120, 180, 240, 300):
            near = min(r1, key=lambda r: abs(float(r["minute"]) - target))
            if abs(float(near["minute"]) - target) < 2:
                print(f"{target:4d} min  {float(near['temp_c']):5.1f} C  throttled={near['throttled']}")
        for p in th:
            rr = list(csv.DictReader(open(p)))
            print(f"{p.stem}: duration {float(rr[-1]['minute']) / 60:.1f} h, max {max(float(r['temp_c']) for r in rr):.1f} C, "
                  f"throttling seen: {any(r['throttled'] not in ('0x0', 'n/a') for r in rr)}")
    for p in sorted(OUT.glob("overnight_run*.csv")):
        rr = list(csv.DictReader(open(p)))
        if rr:
            fps = [float(r["fps"]) for r in rr]
            print(f"\n{p.stem}: {float(rr[-1]['minute']) / 60:.2f} h, {rr[-1]['frames']} frames, "
                  f"FPS mean {np.mean(fps):.1f} (first hour {np.mean(fps[:60]):.1f}, last hour {np.mean(fps[-60:]):.1f})")
    for p in sorted(OUT.glob("conv_overnight_run*.csv")):
        rr = [r for r in csv.DictReader(open(p)) if not r["error"]]
        if rr:
            print(f"{p.stem}: {len(rr)} turns, median T_conv {np.median([float(r['t_conv']) for r in rr]):.0f} ms")
    fp = OUT / "failures.csv"
    if fp.exists():
        f = list(csv.DictReader(open(fp)))
        print(f"\nfailure events logged: {len(f)}")
        for run in sorted({r['run'] for r in f}):
            ev = [r for r in f if r["run"] == run]
            print(f"  run {run}: {len(ev)} events, types: {sorted({r['type'] for r in ev})}")
    print(f"\nfigures written to {figs}/")

def cmd_fp(a):
    import cv2
    hdet = haar()
    ydet = None
    rows = []
    for gt in sorted(FRAMES.glob("*/ground_truth.csv")):
        cond = gt.parent.name
        truth = [r for r in csv.DictReader(open(gt))]
        imgs = [cv2.imread(str(gt.parent / r["file"])) for r in truth]
        h, w = imgs[0].shape[:2]
        if ydet is None:
            ydet = yunet(w, h)
        for name in ("haar", "yunet"):
            tp = fp = fn = nf = nf_alarm = 0
            for r, img in zip(truth, imgs):
                if name == "haar":
                    boxes = [tuple(map(float, b)) for b in hdet.detectMultiScale(
                        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=a.scale, minNeighbors=a.neighbors)]
                else:
                    _, faces = ydet.detect(img)
                    boxes = [] if faces is None else [tuple(map(float, f[:4])) for f in faces]
                if int(r["face"]):
                    ref = tuple(float(r[k]) for k in "xywh")
                    best = max((iou(b, ref) for b in boxes), default=0.0)
                    if best >= 0.5:
                        tp += 1; fp += len(boxes) - 1
                    else:
                        fn += 1; fp += len(boxes)
                else:
                    nf += 1; fp += len(boxes); nf_alarm += 1 if boxes else 0
            prec = 100.0 * tp / (tp + fp) if tp + fp else float("nan")
            rec = 100.0 * tp / (tp + fn) if tp + fn else float("nan")
            rows.append(dict(condition=cond, detector=name, frames=len(imgs), face_frames=tp + fn,
                             tp=tp, fp=fp, fn=fn, precision=round(prec, 1), recall=round(rec, 1),
                             no_face_frames=nf, no_face_frames_with_detection=nf_alarm))
            print(rows[-1])
    if not rows:
        sys.exit("no annotated frames found (measurements/frames/*/ground_truth.csv)")
    with open(OUT / "fp.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader(); wr.writerows(rows)
    print("\n=== totals over all conditions ===")
    for name in ("haar", "yunet"):
        R = [r for r in rows if r["detector"] == name]
        TP, FP, FN = (sum(r[k] for r in R) for k in ("tp", "fp", "fn"))
        NF, NA = sum(r["no_face_frames"] for r in R), sum(r["no_face_frames_with_detection"] for r in R)
        print(f"{name:6s} TP {TP}  FP {FP}  FN {FN}  precision {100*TP/max(TP+FP,1):.1f} %  "
              f"recall {100*TP/max(TP+FN,1):.1f} %  no-face frames {NF}, with a detection {NA}")
    print("saved", OUT / "fp.csv")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="mode", required=True)

    t = sp.add_parser("track")
    t.add_argument("--condition", required=True, help="e.g. bright_1.0m (light_distance)")
    t.add_argument("--frames", type=int, default=600)
    t.add_argument("--save-every", type=int, default=5, help="save every Nth frame for annotation")
    t.add_argument("--width", type=int, default=320)
    t.add_argument("--height", type=int, default=240)
    t.add_argument("--display", action="store_true", help="show the camera window like the robot did (adds its time to T_track)")
    t.add_argument("--no-servo", action="store_true")

    n = sp.add_parser("annotate")
    n.add_argument("--condition")

    d = sp.add_parser("detect")
    d.add_argument("--scale", type=float, default=1.3, help="robot used 1.3")
    d.add_argument("--neighbors", type=int, default=5)

    f_ = sp.add_parser("fp", help="false positives / precision on the annotated frames")
    f_.add_argument("--scale", type=float, default=1.3)
    f_.add_argument("--neighbors", type=int, default=5)

    c = sp.add_parser("conv")
    c.add_argument("--questions", default="questions")
    c.add_argument("--repeats", type=int, default=10)
    c.add_argument("--load", type=int, default=0, help="extra CPU load in %% via stress-ng (0 = none)")
    c.add_argument("--model", default="gpt-4o-mini", help="gpt-3.5-turbo (used on the robot) has been retired by OpenAI")

    h = sp.add_parser("thermal")
    h.add_argument("--hours", type=float, default=5)
    h.add_argument("--interval", type=float, default=10)
    h.add_argument("--run", type=int, default=1)

    o = sp.add_parser("overnight", help="unattended long run (Step 7 + Step 8)")
    o.add_argument("--hours", type=float, default=5)
    o.add_argument("--run", type=int, default=1)
    o.add_argument("--interval", type=float, default=10, help="temperature sample period (s)")
    o.add_argument("--ask-every", type=float, default=5, help="minutes between questions")
    o.add_argument("--questions", default="questions")
    o.add_argument("--model", default="gpt-4o-mini")
    o.add_argument("--no-servo", action="store_true")

    sp.add_parser("report")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    globals()["cmd_" + a.mode](a)

if __name__ == "__main__":
    main()
