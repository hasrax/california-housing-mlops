import hashlib

import pandas as pd


def dataframe_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("|".join(map(str, frame.columns)).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()
