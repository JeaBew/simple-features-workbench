from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple, Optional

from cassis import Cas
from py_lift.dkpro import T_SENT, T_TOKEN, T_DEP
from py_lift.extractors import FEL_BaseExtractor, get_default_extractor_strict

class FE_FreqBandRatios(FEL_BaseExtractor):
    """Extractor that computes the ratio of frequency bands per token. 
    Assumes that 'org.lift.type.Frequency' structure annotations are present in the CAS."""
    
    def __init__(self):
        super().__init__()
    
    def extract(self, cas: Cas) -> bool:
        """Extract frequency‑band ratios, counting each token **once**.

        The original implementation counted each ``Frequency`` annotation
        directly, which could lead to double‑counting when a token had multiple
        overlapping annotations.  This version iterates over the ``Token``
        annotations, determines the associated frequency band for each token
        (falling back to ``oov`` when none is present), and then computes the
        ratios based on the unique token count.
        """
        F = self.ts.get_type("FeatureAnnotationNumeric")

        # Map each token span to its frequency band (or ``oov``).
        token_band_counts: Counter[str] = Counter()
        # Build a lookup of frequency annotations by span for fast access.
        freq_by_span: dict[Tuple[int, int], str] = {}
        for freq in cas.select('org.lift.type.Frequency'):
            band = freq.get('frequencyBand')
            if band is None:
                continue
            # Store using the annotation's begin/end as the key.
            freq_by_span[(int(freq.begin), int(freq.end))] = str(band).strip().lower()

        # Iterate over tokens and assign a band.
        for token in cas.select(T_TOKEN):
            span = (int(token.begin), int(token.end))
            band = freq_by_span.get(span, 'oov')
            token_band_counts[band] += 1

        # Handle empty case (no tokens).
        total = sum(token_band_counts.values())
        if total == 0:
            return True

        # Define feature mappings.
        feature_mappings = {
            'f1': 'Freq_Ratio_F1_PER_Token',
            'f2': 'Freq_Ratio_F2_PER_Token',
            'f3': 'Freq_Ratio_F3_PER_Token',
            'f4': 'Freq_Ratio_F4_PER_Token',
            'f5': 'Freq_Ratio_F5_PER_Token',
            'f6': 'Freq_Ratio_F6_PER_Token',
            'f7': 'Freq_Ratio_F7_PER_Token',
            'oov': 'Freq_Ratio_OOV_PER_Token',
        }

        # Create and add features.
        for band, feature_name in feature_mappings.items():
            ratio = token_band_counts.get(band, 0) / total
            feature = F(name=feature_name, value=ratio, begin=0, end=0)
            cas.add(feature)

        return True