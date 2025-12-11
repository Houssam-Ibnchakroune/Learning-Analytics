import psycopg2
import os
conn=psycopg2.connect(
        host     = os.getenv("PG_HOST", "localhost"),
        port     = int(os.getenv("PG_PORT", 5432)),
        dbname   = os.getenv("PG_DBNAME", "projet_spark"),
        user     = os.getenv("PG_USER", "postgres"),
        password = os.getenv("PG_PASSWORD", "0000")
    )
print("Connection to PostgreSQL database established successfully.")
ddl="""
select * from customers;
"""
cur = conn.cursor()
cur.execute(ddl)
print(cur.fetchall())
cur.close()
conn.close()