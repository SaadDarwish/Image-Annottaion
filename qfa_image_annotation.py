
"""
QFA-based Multi-Label Image Annotation System
--------------------------------------------
This module implements the proposed model described in the paper:
'Visual Big Data Mining: Toward Next-Generation Multi-Label Image Annotation
and Retrieval Using Quantum Firefly Optimization'

Author: Saad M. Darwish et al.
Reference Implementation (Research/Educational Use)
"""

import numpy as np
import cv2
from skimage import measure
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# =====================================================
# STEP 1: MULTI-LEVEL OTSU THRESHOLDING
# =====================================================

def multilevel_otsu(gray, num_thresholds=2):
    """
    Compute multi-level Otsu thresholds using histogram variance maximization.
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()

    def between_class_variance(thresholds):
        thresholds = [0] + list(thresholds) + [255]
        mu_total = sum(i * hist[i] for i in range(256))
        sigma = 0.0
        for i in range(len(thresholds) - 1):
            start, end = thresholds[i], thresholds[i + 1]
            w = sum(hist[start:end + 1])
            if w > 0:
                mu = sum(j * hist[j] for j in range(start, end + 1)) / w
                sigma += w * (mu - mu_total) ** 2
        return sigma

    best_sigma = -1
    best_thresholds = None

    for t1 in range(50, 200, 20):
        for t2 in range(t1 + 10, 240, 20):
            thresholds = [t1, t2][:num_thresholds]
            sigma = between_class_variance(thresholds)
            if sigma > best_sigma:
                best_sigma = sigma
                best_thresholds = thresholds

    return best_thresholds


# =====================================================
# STEP 2: QUANTUM FIREFLY OPTIMIZATION
# =====================================================

def levy_flight(mu=1.5):
    u = np.random.normal(0, 1)
    v = np.random.normal(0, 1)
    return u / (abs(v) ** (1 / mu))


def qfa_optimize_thresholds(gray, otsu_thresholds, n_fireflies=15, iterations=20):
    m = len(otsu_thresholds)
    fireflies = np.array([
        otsu_thresholds + np.random.uniform(-5, 5, m)
        for _ in range(n_fireflies)
    ])

    beta0, gamma, alpha = 1.0, 1.0, 0.2

    def fitness(thresholds):
        thresholds = np.clip(np.sort(thresholds), 0, 255).astype(int)
        segmented = np.digitize(gray, thresholds)
        return np.var(segmented)

    for _ in range(iterations):
        brightness = np.array([fitness(f) for f in fireflies])
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if brightness[j] > brightness[i]:
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    fireflies[i] += (
                        beta0 * np.exp(-gamma * r ** 2) *
                        (fireflies[j] - fireflies[i]) +
                        alpha * levy_flight()
                    )

    best = fireflies[np.argmax([fitness(f) for f in fireflies])]
    return np.clip(np.sort(best), 0, 255).astype(int)


# =====================================================
# STEP 3: SEGMENTATION AND MORPHOLOGY
# =====================================================

def segment_image(gray, thresholds):
    return np.digitize(gray, thresholds)


def remove_small_regions(label_img, min_area=100):
    cleaned = np.zeros_like(label_img)
    for region in measure.regionprops(label_img):
        if region.area >= min_area:
            cleaned[label_img == region.label] = region.label
    return cleaned


# =====================================================
# STEP 4: FEATURE EXTRACTION
# =====================================================

def extract_blob_features(image, label_img):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    features = []

    for region in measure.regionprops(label_img):
        mask = label_img == region.label
        pixels = lab[mask]

        mean = pixels.mean(axis=0)
        std = pixels.std(axis=0)
        skew = np.mean((pixels - mean) ** 3, axis=0)

        area = region.area
        perimeter = region.perimeter
        convexity = region.area / region.convex_area if region.convex_area > 0 else 0

        feature = np.concatenate([mean, std, skew, [area, perimeter, convexity]])
        features.append(feature)

    return np.array(features)


def reduce_features(features, target_dim=12):
    if features.shape[1] <= target_dim:
        return features
    pca = PCA(n_components=target_dim)
    return pca.fit_transform(features)


# =====================================================
# STEP 5: BAYESIAN TRAINING
# =====================================================

def train_bayesian_model(feature_sets, label_sets, num_clusters=10):
    all_features = np.vstack(feature_sets)
    gmm = GaussianMixture(n_components=num_clusters, covariance_type='full')
    gmm.fit(all_features)

    label_probs = {}
    for features, labels in zip(feature_sets, label_sets):
        clusters = gmm.predict(features)
        for c in clusters:
            label_probs.setdefault(c, {})
            for label in labels:
                label_probs[c][label] = label_probs[c].get(label, 0) + 1

    for c in label_probs:
        total = sum(label_probs[c].values())
        for label in label_probs[c]:
            label_probs[c][label] /= total

    return gmm, label_probs


# =====================================================
# STEP 6: AUTO-ANNOTATION
# =====================================================

def annotate_image(image, gmm, label_probs):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    otsu_t = multilevel_otsu(gray)
    qfa_t = qfa_optimize_thresholds(gray, otsu_t)

    labels = segment_image(gray, qfa_t)
    labels = remove_small_regions(labels)

    features = extract_blob_features(image, labels)
    features = reduce_features(features)

    annotations = {}
    clusters = gmm.predict(features)

    for c in clusters:
        if c in label_probs:
            for label, prob in label_probs[c].items():
                annotations[label] = annotations.get(label, 0) + prob

    return sorted(annotations.items(), key=lambda x: x[1], reverse=True)


# =====================================================
# MAIN PIPELINE
# =====================================================

def qfa_annotation_pipeline(train_images, train_labels, test_image):
    feature_sets = []

    for img in train_images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        otsu_t = multilevel_otsu(gray)
        qfa_t = qfa_optimize_thresholds(gray, otsu_t)
        labels = segment_image(gray, qfa_t)
        labels = remove_small_regions(labels)
        features = extract_blob_features(img, labels)
        features = reduce_features(features)
        feature_sets.append(features)

    gmm, label_probs = train_bayesian_model(feature_sets, train_labels)
    return annotate_image(test_image, gmm, label_probs)
