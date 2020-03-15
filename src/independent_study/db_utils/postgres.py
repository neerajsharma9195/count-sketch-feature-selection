import psycopg2
import json
POSTGRE_DB_NAME = "analysisdb"
POSTGRE_USERNAME = "neerajsharma"
POSTGRE_HOST = "127.0.0.1"
POSTGRE_PASSWORD = ""


def check_connection():
    try:
        conn = psycopg2.connect(
            "dbname={} user={} host={} password={}".format(POSTGRE_DB_NAME, POSTGRE_USERNAME, POSTGRE_HOST,
                                                           POSTGRE_PASSWORD))
        print(conn)
        conn.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def insert_bulk(table_name, tuple_array):
    '''

    :param tuple_array: An array of tuples which need to be inserted in review_feedback table
    :return:
    '''
    try:
        conn = psycopg2.connect(
            "dbname={} user={} host={} password={}".format(POSTGRE_DB_NAME, POSTGRE_USERNAME, POSTGRE_HOST,
                                                           POSTGRE_PASSWORD))
        insert_query = "INSERT INTO {} VALUES (%s, %s)".format(table_name)
        cur = conn.cursor()
        cur.executemany(insert_query, tuple_array)
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error, 'exception in inserting in bulk amount postgre ' + str(error))
    finally:
        if conn is not None:
            conn.close()


def fetch_data(file_path):
    tuples = []
    with open(file_path, 'r') as f:
        for line in f:
            pos, val = line.split(":")
            pos = int(pos)
            val = float(val)
            tuples.append((pos, val))
    return tuples


def fetch_data_for_kanchi(file_path):
    tuples = []
    with open(file_path, 'r') as f:
        data = json.loads(f.read())
        for key, value in data.items():
            tuples.append((int(key), float(value)))
    return tuples


file_path = "../dumps/top8000_logistic_regression2020-03-08 16:19:19.250246.json"
data_tuples = fetch_data_for_kanchi(file_path)
insert_bulk('top_k_logistic', data_tuples)

# check_connection()
