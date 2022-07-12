import mysql.connector

def insert(thePath, theAfimi):
    path = thePath
    afimi = theAfimi
    try:
        connection = mysql.connector.connect(host="localhost",
                                            database='signatures',
                                            user="root",
                                            password="")

        mySql_insert_query = """INSERT INTO signat (afimi, sig) 
                            VALUES 
                            (%s, %s) """

        cursor = connection.cursor()
        cursor.execute(mySql_insert_query, (str(afimi),path))
        connection.commit()
        print(cursor.rowcount, "Record inserted successfully into signatures table")
        cursor.close()

    except mysql.connector.Error as error:
        print("Failed to insert record into signatures table {}".format(error))

    finally:
        if connection.is_connected():
            connection.close()
            print("MySQL connection is closed")