"""
Perceptual Image Hashing for Hot-Path Image Dedup.

Implements dHash (difference hash) and pHash (DCT-based) for detecting
visually similar images — critical for known CSAM/illegal imagery matching.

Algorithm (dHash):
  1. Resize to 9×8 grayscale
  2. Compute horizontal gradient (diff between adjacent pixels)
  3. Hash = 64-bit fingerprint (1 if right > left, else 0)
  4. Hamming distance < threshold → visually similar

Why dHash for POC:
  - Fast (< 1ms for thumbnail-sized images, pure Python + Pillow)
  - Robust to: resizing, slight color changes, mild compression
  - Not robust to: heavy cropping, rotation, mirroring (need PhotoDNA/PDQ for that)

Production upgrade path:
  - Facebook PDQ (pip install pdqhash) — DCT-based, more robust
  - Microsoft PhotoDNA — requires license, gold standard for CSAM
  - Apple NeuralHash — neural-network based, most robust to heavy edits

Reference hashes:
  In production, the hash database comes from authoritative sources
  (NCMEC, Interpol, national law enforcement). POC uses synthetic
  test data only — no real illegal imagery or hashes.

Usage:
    from src.skills.image_phash import image_phash
    h = image_phash.dhash(image_bytes)       # 64-bit hex hash
    dist = image_phash.hamming(h1, h2)       # bit distance (< 10 = similar)
"""

import io
import logging
from PIL import Image

logger = logging.getLogger(__name__)

# Hamming distance threshold: hashes with distance ≤ this are considered matches.
# dHash on 64 bits at threshold=10: ~0.001% false positive rate.
HAMMING_THRESHOLD = 10


class ImagePHash:
    """Perceptual hash for image deduplication and known-content matching."""

    def __init__(self, hash_size: int = 8):
        """
        Args:
            hash_size: dHash size. 8 → 64-bit hash. 16 → 256-bit (more precise).
        """
        self.hash_size = hash_size
        self._known_hashes: dict[str, dict] = {}  # hex_hash → metadata

    # -- dHash: difference hash --

    def dhash(self, image_bytes: bytes) -> str:
        """Compute 64-bit dHash for image bytes.

        Returns hex string like "a3f7c9b01e4d82f6".
        Returns empty string if image cannot be decoded.
        """
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("L")
            # Resize to (hash_size+1) × hash_size
            img = img.resize((self.hash_size + 1, self.hash_size), Image.LANCZOS)
        except Exception as e:
            logger.warning("dHash: cannot decode image: %s", e)
            return ""

        pixels = list(img.getdata())

        # Compute horizontal difference hash
        hash_bits = []
        for row in range(self.hash_size):
            for col in range(self.hash_size):
                left = pixels[row * (self.hash_size + 1) + col]
                right = pixels[row * (self.hash_size + 1) + col + 1]
                hash_bits.append("1" if left < right else "0")

        # Convert binary string to hex
        hash_int = int("".join(hash_bits), 2)
        return format(hash_int, f"0{self.hash_size * self.hash_size // 4}x")

    # -- pHash: DCT-based hash (more robust to resize/compression) --

    def phash(self, image_bytes: bytes, highfreq_factor: int = 4) -> str:
        """Compute DCT-based pHash. More robust than dHash for resized images.

        Requires numpy + scipy (optional dependencies).
        Falls back to dHash if unavailable.
        """
        try:
            import numpy as np
        except ImportError:
            logger.debug("pHash: numpy not available, falling back to dHash")
            return self.dhash(image_bytes)

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("L")
            img_size = self.hash_size * highfreq_factor
            img = img.resize((img_size, img_size), Image.LANCZOS)
        except Exception as e:
            logger.warning("pHash: cannot decode image: %s", e)
            return ""

        pixels = np.array(img, dtype=np.float32)

        # DCT-II on 2D array
        dct = self._dct2d(pixels)

        # Keep low-frequency top-left corner
        dct_low = dct[:self.hash_size, :self.hash_size]

        # Median as threshold
        median = np.median(dct_low)

        # Bits: 1 if above median
        hash_bits = (dct_low > median).flatten()
        hash_str = "".join("1" if b else "0" for b in hash_bits)

        hash_int = int(hash_str, 2)
        return format(hash_int, f"0{self.hash_size * self.hash_size // 4}x")

    @staticmethod
    def _dct2d(arr: "np.ndarray") -> "np.ndarray":
        """Simple 2D DCT (no scipy dependency)."""
        import numpy as np
        N = arr.shape[0]
        dct = np.zeros_like(arr)
        for u in range(N):
            for v in range(N):
                total = 0.0
                for x in range(N):
                    for y in range(N):
                        total += (
                            arr[x, y]
                            * np.cos((2 * x + 1) * u * np.pi / (2 * N))
                            * np.cos((2 * y + 1) * v * np.pi / (2 * N))
                        )
                dct[u, v] = total * (1 / np.sqrt(2) if u == 0 else 1) * (1 / np.sqrt(2) if v == 0 else 1) * (2 / N)
        return dct

    # -- Hamming distance --

    @staticmethod
    def hamming(h1: str, h2: str) -> int:
        """Hamming distance between two hex hash strings."""
        if not h1 or not h2 or len(h1) != len(h2):
            return 999  # max distance for invalid/mismatched hashes
        # Convert hex to ints, XOR, count bits
        i1 = int(h1, 16)
        i2 = int(h2, 16)
        return (i1 ^ i2).bit_count()

    # -- Known hash database --

    def load_known_hashes(self, hash_dict: dict[str, dict]):
        """Load a database of known illegal image hashes.

        Format: {"hash_hex": {"category": "csam", "source": "ncmec", ...}, ...}
        """
        self._known_hashes.update(hash_dict)
        logger.info("Loaded %d known hashes", len(hash_dict))

    def check_known(self, image_hash: str) -> dict | None:
        """Check if a hash matches any known illegal image hash.

        Returns metadata dict if Hamming distance ≤ threshold, else None.
        """
        if not image_hash:
            return None
        for known_hash, metadata in self._known_hashes.items():
            if self.hamming(image_hash, known_hash) <= HAMMING_THRESHOLD:
                return {**metadata, "matched_hash": known_hash}
        return None


# Singleton
image_phash = ImagePHash()

# --- POC known-hash database (synthetic test data only) ---
# Format: {"hash_hex": {"category": str, "action": "block", "source": str}}
# In production, this comes from authoritative sources (NCMEC, Interpol, etc.)
# and may contain millions of entries stored in a specialized database.
#
# These are hashes of synthetically generated solid-color images — NOT real
# illegal imagery. They exist solely to verify the matching pipeline works.
_SYNTHETIC_KNOWN = {
    # These will be populated by the benchmark/test script with hashes
    # of actual test images. Empty in production unless loaded externally.
}
if _SYNTHETIC_KNOWN:
    image_phash.load_known_hashes(_SYNTHETIC_KNOWN)
