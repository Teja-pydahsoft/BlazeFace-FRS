import os
import tempfile
import numpy as np
from app.core.database import DatabaseManager


def test_add_and_get_face_encodings_roundtrip(tmp_path):
    db_file = tmp_path / "test_db.db"
    dm = DatabaseManager(str(db_file))

    # Create a dummy student
    student_data = {'student_id': 's100', 'name': 'Test Student'}
    assert dm.add_student(student_data) is True

    # Create synthetic encodings
    enc1 = np.random.rand(128).astype('float32')
    enc2 = np.random.rand(128).astype('float32')

    # Add encodings
    assert dm.add_face_encodings('s100', [enc1, enc2], encoding_type='testembed', image_paths=[None, None]) is True

    # Read back
    encs = dm.get_face_encodings('s100')
    assert len(encs) == 2
    for sid, enc, enc_type, img in encs:
        assert sid == 's100'
        assert enc_type == 'testembed'
        assert isinstance(enc, np.ndarray)
        assert enc.dtype == np.float32
        assert enc.shape[0] == 128
