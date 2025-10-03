import numpy as np
from app.utils.encoding_quality_checker import EncodingQualityChecker


def test_quality_checker_accepts_good_encoding():
    checker = EncodingQualityChecker()
    # Create a realistic encoding
    base = np.random.rand(128).astype('float32')
    base = base / np.linalg.norm(base)

    # Existing encodings (same person) similar to base
    # Create an existing encoding very close to base (normalized)
    noise = np.random.normal(0, 0.005, size=base.shape).astype('float32')
    existing_enc = base + noise
    existing_enc = existing_enc / np.linalg.norm(existing_enc)
    existing = [existing_enc]

    # Other people's encodings (dissimilar)
    # Other people's encodings should be dissimilar
    # Create an other-person encoding that's orthogonal (as much as possible) to base to ensure low similarity
    other_vec = np.random.rand(128).astype('float32')
    # Remove projection onto base to make it dissimilar
    proj = np.dot(other_vec, base) * base
    other_vec = other_vec - proj
    if np.linalg.norm(other_vec) == 0:
        # fallback: perturb base and invert
        other_vec = np.roll(base, 1) * -1.0
    other_vec = other_vec / np.linalg.norm(other_vec)
    other = [('sX', other_vec)]

    ok, reason, metrics = checker.check_new_encoding_quality(base, existing, other, 's100')
    assert ok is True, f"Quality checker unexpectedly rejected encoding: {reason}"
    assert 'quality_score' in metrics
    assert metrics['quality_score'] >= 0.0


def test_quality_checker_rejects_duplicate_of_other():
    checker = EncodingQualityChecker()
    base = np.random.rand(128).astype('float32')
    base /= np.linalg.norm(base)

    # Other's encoding very similar -> should trigger duplicate detection
    other_similar = [('sY', base * 0.99)]
    ok, reason, metrics = checker.check_new_encoding_quality(base, [], other_similar, 's101')
    assert ok is False
    assert 'Potential duplicate' in reason or 'Too similar to other people' in reason
