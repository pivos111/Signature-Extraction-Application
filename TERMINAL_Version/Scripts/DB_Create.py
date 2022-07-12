import mysql.connector
try:
    connection = mysql.connector.connect(host="localhost",
                                        user="root",
                                        password="")

    mySql_insert_query = """CREATE DATABASE test"""
    cursor = connection.cursor()
    cursor.execute(mySql_insert_query)
    connection.commit()
    print(cursor.rowcount, "DB successfully created")
    cursor.close()

except mysql.connector.Error as error:
    print("Failed to insert record into signatures table {}".format(error))

finally:
    if connection.is_connected():
        connection.close()
        print("MySQL connection is closed")

try:
    connection = mysql.connector.connect(host="localhost",
                                         database='test',
                                         user="root",
                                         password="")

    mySql_insert_query = """CREATE TABLE signat (afimi INT(11) PRIMARY KEY, sig TEXT)"""
    cursor = connection.cursor()
    cursor.execute(mySql_insert_query)
    connection.commit()
    print(cursor.rowcount, "Table successfully created")
    cursor.close()

except mysql.connector.Error as error:
    print("Failed to insert record into signatures table {}".format(error))

finally:
    if connection.is_connected():
        connection.close()
        print("MySQL connection is closed")