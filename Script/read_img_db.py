import pyodbc
import os
import io
import PIL.Image as Image

conn = pyodbc.connect(r'Driver=SQL Server;Server=.\SQLN;Database=hrm_police;Trusted_Connection=yes;')
cursor = conn.cursor()

query = """
SELECT [nom] + '_' + [postnom] + '_' + [matricule]
      ,[photo]
  FROM [hrm_police].[dbo].[agent]
"""
imagePath = 'temp_img'
try:
    cursor = conn.cursor()
    cursor.execute(query)
    
    while True: 
        row = cursor.fetchone()
        if row is None:
            break
        filename = os.path.join(imagePath, row[0] + ".jpg")
        print("Running for ", row[0])
        image = Image.open(io.BytesIO(row[1]))
        image.save(filename)

except Exception as error:
    print(error)
finally:
    cursor.close()
    conn.close()