import pandas as pd
import sqlite3

conn = sqlite3.connect('insurance_info.db')

df = pd.read_csv('datapkg/마케팅정보.csv', encoding='utf-8')
df.columns = ['ins_type', 'secondary_type', 'ins_name', 'ad_text', 'benefits', 'necessity']
df.to_sql('insurance_info', conn, if_exists='replace', index=False)

conn.close()


