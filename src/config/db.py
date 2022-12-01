import mysql.connector as sql

# Variabile per la connessione al DB temporaneamente settata a None
db_connection = None
datasets_conn = None
social_crawler_conn = None

# Credenziali per la connessione al DB
'''localhost
db_host = 'localhost'
db_user = 'root'
db_password = ''
'''

#db_host = '192.167.149.145'
#db_user = 'root'
#db_password = '7911qgtr5491'

db_host = 'localhost'
db_user = 'root'
db_password = ''


datasets_conn = sql.connect(host=db_host, database="dataset_tmp", user=db_user, password=db_password)

#social_crawler_conn = sql.connect(host=db_host, database="social_crawler", user=db_user, password=db_password)
