import glob
import json
import os
import random
import time

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (auc, average_precision_score, precision_recall_curve, roc_curve)
from sklearn.svm import SVC, LinearSVC
from FaceDetector import WIN, detect, draw_faces, make_hog


SEED = 0
LFW_GLOB = "lfw/*/*.jpg"
NEG_DIRS = ["ade", "caltech"]
FACE_MARGIN = 0.244          # crop 61:189 of a 250x250 LFW image (the head area)
N_TRAIN, N_VAL, N_TEST = 10000, 1000, 1000
N_TOTAL = N_TRAIN + N_VAL + N_TEST
CACHE_NPZ = "dataset_cache.npz"
CACHE_SRC = "train_neg_sources.json"

# (cellSize, blockSize, blockStride) - block must be a multiple of cell
HOG_CONFIGS = [(8, 16, 8), (8, 32, 8), (16, 32, 16), (4, 8, 4)]
MINING_IMAGES = 200          # negative images used for hard-negative mining
MAX_HARD = 20000
TEST_IMAGES = [("apollo.jpg", "res4.jpg")]


def build_faces(rng):
    files = sorted(glob.glob(LFW_GLOB))
    if len(files) < N_TOTAL:
        raise SystemExit("Found only %d LFW images with '%s' (need %d)."
                         % (len(files), LFW_GLOB, N_TOTAL))
    rng.shuffle(files)
    faces = []
    for f in files[:N_TOTAL]:
        img = cv2.imread(f, 0)
        h, w = img.shape
        m = int(round(FACE_MARGIN * h))
        crop = img[m:h - m, m:w - m]
        faces.append(cv2.resize(crop, (WIN, WIN), interpolation=cv2.INTER_AREA))
    return np.array(faces, np.uint8)


def negative_sources():
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files, skipped = [], 0
    for d in NEG_DIRS:
        for p in glob.glob(os.path.join(d, "**", "*"), recursive=True):
            if not p.lower().endswith(exts):
                continue
            if "faces" in p.lower():
                skipped += 1
                continue
            files.append(p)
    print("negative source images: %d (skipped %d in 'faces' folders)" % (len(files), skipped))
    if len(files) < 30:
        raise SystemExit("Too few negative source images. Check NEG_DIRS.")
    return files


def make_patches(files, n, rng):
    out, tries = [], 0
    while len(out) < n:
        tries += 1
        if tries > n * 20:
            raise SystemExit("Not enough usable negative images.")
        img = cv2.imread(rng.choice(files), 0)
        if img is None or min(img.shape) < WIN:
            continue
        for _ in range(5):                      # 5 patches per image read
            if len(out) >= n:
                break
            h, w = img.shape
            s = rng.uniform(0.3, 1.0)           # random scale, like pyramid levels
            if min(h, w) * s >= WIN:
                cur = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            else:
                cur = img
            ch, cw = cur.shape
            y, x = rng.randint(0, ch - WIN), rng.randint(0, cw - WIN)
            out.append(cur[y:y + WIN, x:x + WIN])
    return np.array(out, np.uint8)


def build_negatives(rng):
    """Split the SOURCE images 80/10/10 first, then crop patches, so no source
    image is shared between train, validation and test."""
    files = negative_sources()
    rng.shuffle(files)
    n = len(files)
    tr, va, te = files[:int(.8 * n)], files[int(.8 * n):int(.9 * n)], files[int(.9 * n):]
    negs = np.concatenate([make_patches(tr, N_TRAIN, rng),
                           make_patches(va, N_VAL, rng),
                           make_patches(te, N_TEST, rng)])
    return negs, tr


def get_dataset():
    if os.path.exists(CACHE_NPZ) and os.path.exists(CACHE_SRC):
        d = np.load(CACHE_NPZ)
        with open(CACHE_SRC) as f:
            return d["faces"], d["negs"], json.load(f)
    rng = random.Random(SEED)
    print("building face crops ...")
    faces = build_faces(rng)
    print("building negative patches ...")
    negs, train_src = build_negatives(rng)
    np.savez_compressed(CACHE_NPZ, faces=faces, negs=negs)
    with open(CACHE_SRC, "w") as f:
        json.dump(train_src, f)
    return faces, negs, train_src


def features(hog, imgs):
    return np.array([hog.compute(im).ravel() for im in imgs], np.float32)


def split_xy(hog, faces, negs, a, b):
    X = np.vstack([features(hog, faces[a:b]), features(hog, negs[a:b])])
    y = np.r_[np.ones(b - a), np.zeros(b - a)].astype(int)
    return X, y


def main(skip_rbf):
    t0 = time.time()
    faces, negs, train_src = get_dataset()
    print("faces", faces.shape, "negatives", negs.shape)
    tr = (0, N_TRAIN)
    va = (N_TRAIN, N_TRAIN + N_VAL)
    te = (N_TRAIN + N_VAL, N_TOTAL)
    log = []

    # choose HOG parameters (fast linear SVM as the probe)
    best = (-1, None)
    for cfg in HOG_CONFIGS:
        hog = make_hog(*cfg)
        Xtr, ytr = split_xy(hog, faces, negs, *tr)
        Xva, yva = split_xy(hog, faces, negs, *va)
        clf = LinearSVC(C=0.1, dual=False, max_iter=5000).fit(Xtr, ytr)
        acc = clf.score(Xva, yva)
        line = "HOG cell=%d block=%d stride=%d dim=%d  val acc=%.4f" % (
            cfg[0], cfg[1], cfg[2], Xtr.shape[1], acc)
        print(line)
        log.append(line)
        if acc > best[0]:
            best = (acc, cfg)
    cfg = best[1]
    log.append("BEST HOG: cell=%d block=%d blockStride=%d nbins=9" % cfg)
    print(log[-1])

    hog = make_hog(*cfg)
    Xtr, ytr = split_xy(hog, faces, negs, *tr)
    Xva, yva = split_xy(hog, faces, negs, *va)
    Xte, yte = split_xy(hog, faces, negs, *te)

    # choose kernel and parameters on the validation set
    cands = [("linear C=%g" % C, LinearSVC(C=C, dual=False, max_iter=5000))
             for C in (0.01, 0.1, 1, 10)]
    if not skip_rbf:
        g0 = 1.0 / (Xtr.shape[1] * Xtr.var())
        cands += [("rbf C=%g gamma=%.3g" % (C, g0 * m), SVC(kernel="rbf", C=C, gamma=g0 * m))
                  for C in (1, 10) for m in (0.5, 1, 2)]
        cands += [("poly deg=3 C=1", SVC(kernel="poly", degree=3, C=1, gamma=g0))]
    best_acc, best_clf, best_desc = -1, None, ""
    for desc, clf in cands:
        t = time.time()
        clf.fit(Xtr, ytr)
        acc = clf.score(Xva, yva)
        line = "SVM %-28s val acc=%.4f  (%.0fs)" % (desc, acc, time.time() - t)
        print(line)
        log.append(line)
        if acc > best_acc:
            best_acc, best_clf, best_desc = acc, clf, desc
    log.append("BEST SVM: %s  val acc=%.4f" % (best_desc, best_acc))
    log.append("TEST acc (before hard-negative mining): %.4f" % best_clf.score(Xte, yte))
    print(log[-2]); print(log[-1])

    # hard-negative mining on TRAIN negative source images
    rng = random.Random(SEED + 1)
    tmp_model = {"clf": best_clf, "hog": cfg, "thr": 0.0}
    hard = []
    srcs = rng.sample(train_src, min(MINING_IMAGES, len(train_src)))
    print("mining hard negatives from %d images ..." % len(srcs))
    for k, f in enumerate(srcs):
        img = cv2.imread(f)
        if img is None or min(img.shape[:2]) < WIN:
            continue
        s = 800.0 / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        _, _, fe = detect(img, tmp_model, thr=0.0, scale_step=1.3, stride=16, return_feats=True)
        hard += fe
    if len(hard) > MAX_HARD:
        hard = rng.sample(hard, MAX_HARD)
    print("hard negatives collected:", len(hard))
    if hard:
        Xh = np.vstack([Xtr, np.array(hard, np.float32)])
        yh = np.concatenate([ytr, np.zeros(len(hard), int)])
        final = clone(best_clf).fit(Xh, yh)
    else:
        final = best_clf
    log.append("After mining (%d hard negatives): val acc=%.4f  TEST acc=%.4f" % (
        len(hard), final.score(Xva, yva), final.score(Xte, yte)))
    print(log[-1])

    model = {"clf": final, "hog": cfg, "thr": 0.5, "desc": best_desc}
    joblib.dump(model, "model.pkl")

    # ROC / PR / AP on the test set
    scores = final.decision_function(Xte)
    fpr, tpr, _ = roc_curve(yte, scores)
    plt.figure()
    plt.plot(fpr, tpr, label="AUC = %.4f" % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC"); plt.legend(); plt.grid(True)
    plt.savefig("res1.jpg", dpi=150); plt.close()

    prec, rec, _ = precision_recall_curve(yte, scores)
    ap = average_precision_score(yte, scores)
    plt.figure()
    plt.plot(rec, prec, label="AP = %.4f" % ap)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall"); plt.legend(); plt.grid(True)
    plt.savefig("res2.jpg", dpi=150); plt.close()
    log.append("AP = %.4f   AUC = %.4f" % (ap, auc(fpr, tpr)))
    print(log[-1])

    os.makedirs("debug", exist_ok=True)
    for src, dst in TEST_IMAGES:
        img = cv2.imread(src)
        if img is None:
            print("missing", src, "- skipped")
            continue
        cv2.imwrite(dst, draw_faces(img, model))
        for thr in (0.0, 1.0, 1.5):   # extra outputs to help you pick a threshold
            cv2.imwrite("debug/%s_thr%.1f.jpg" % (os.path.splitext(src)[0], thr),
                        draw_faces(img, model, thr=thr))
        print("saved", dst)

    with open("results.txt", "w") as f:
        f.write("\n".join(log))
    print("done in %.0f s. Summary written to results.txt" % (time.time() - t0))

