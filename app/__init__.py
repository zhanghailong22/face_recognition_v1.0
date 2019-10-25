from .api import face_upload, face_find, video_find,  refresh_redis, face_compare
import psycopg2
conn = psycopg2.connect(host="10.97.97.98", user="postgres", password="root", database="face_images", port=5432)  # connect postgresql
if conn is not None:
    cur = conn.cursor()
    cur.execute("select exists(select * from information_schema.tables where table_name=%s)",('image_data',))
    if cur.fetchone()[0]==0:
        command = "create table if not exists image_data (name text PRIMARY KEY NOT NULL, image bytea NOT NULL);"  # create table cataract
        cur.execute(command)
        r_command = "create rule  r_image_data as on insert to image_data where exists (select 1 from image_data where name =new.name) do instead nothing;"
        cur.execute(r_command)
        conn.commit()  # commit the changes
    cur.close()  # close communication with the PostgreSQL database server
    conn.close()
