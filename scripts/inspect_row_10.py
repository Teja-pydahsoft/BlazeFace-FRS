import sqlite3, os, sys

DB='database/blazeface_frs.db'
conn=sqlite3.connect(DB)
cur=conn.cursor()
cur.execute("PRAGMA table_info(face_encodings)")
cols=cur.fetchall()
print('cols:')
for c in cols:
    print('  ', c)

cur.execute('SELECT * FROM face_encodings WHERE id=10')
row=cur.fetchone()
print('\nrow:')
print(row)
if not row:
    sys.exit(0)
# Map columns
col_names=[c[1] for c in cols]
print('\ncolumn names:')
print(col_names)
row_dict=dict(zip(col_names, row))
print('\nrow as dict:')
print(row_dict)
path=row_dict.get('image_path') or row_dict.get('image') or row_dict.get('photo')
print('\nimage_path:', path)
if path:
    print('exists:', os.path.exists(path))
    if os.path.exists(path):
        print('size:', os.path.getsize(path))
        try:
            from PIL import Image
            im=Image.open(path)
            print('image format, size, mode:', im.format, im.size, im.mode)
        except Exception as e:
            print('PIL open failed:', e)
