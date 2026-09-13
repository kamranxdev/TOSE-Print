"""SOCOFing subject-disjoint dataset protocol and metadata parser."""

import os
import re
from typing import List, Dict, Tuple
from tose_print.core.types import FingerprintSample


class DatasetProtocol:
    """Parses SOCOFing filenames and establishes subject-disjoint partitions."""

    FILENAME_REGEX = re.compile(
        r"^(?P<id>\d+)__(?P<gender>[MF])_(?P<hand>Left|Right)_(?P<finger>\w+)_finger(?:_(?P<alt>CR|Obl|Zcut))?\.BMP$",
        re.IGNORECASE
    )

    @classmethod
    def parse_filename(cls, filename: str) -> Dict[str, any]:
        m = cls.FILENAME_REGEX.match(os.path.basename(filename))
        if not m:
            # Generic fallback
            return {
                "subject_id": 0,
                "gender": "Unknown",
                "hand": "Unknown",
                "finger": "Unknown",
                "alteration": "Real"
            }
        d = m.groupdict()
        return {
            "subject_id": int(d["id"]),
            "gender": d["gender"],
            "hand": d["hand"],
            "finger": d["finger"],
            "alteration": d.get("alt") or "Real"
        }

    @classmethod
    def partition_subjects(
        cls,
        subject_ids: List[int],
        train_ratio: float = 0.33,
        seed: int = 42
    ) -> Tuple[List[int], List[int]]:
        """Splits unique subject IDs into disjoint Train/Calibration and Test sets."""
        import random
        unique_ids = sorted(list(set(subject_ids)))
        rng = random.Random(seed)
        rng.shuffle(unique_ids)

        split_idx = int(len(unique_ids) * train_ratio)
        train_ids = sorted(unique_ids[:split_idx])
        test_ids = sorted(unique_ids[split_idx:])
        return train_ids, test_ids
