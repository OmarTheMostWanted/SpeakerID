import os
import re
import pyodbc
import shutil


def test_db_connection(connectionstring: str):
    try:
        connection = pyodbc.connect(connectionstring)
        cursor = connection.cursor()
        cursor.execute("SELECT @@VERSION;")
        row = cursor.fetchone()
        print(row[0])
        cursor.close()
        connection.close()
        print("Test successful")

    except pyodbc.Error as e:
        print(f"Error connecting to database: {e}")
        print("SQLSTATE: ", e.args[0])
        print("Message: ", e.args[1])
        return False
    return True


def test_db_connection2(connectionstring: str):
    os.environ['OPENSSL_CONF'] = '/home/tmw/.openssl_allow_tls1.0.cnf'

    from sqlalchemy import create_engine, text

    import pypyodbc as odbc

    from sqlalchemy.engine import URL

    # create engine
    connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connectionstring})
    engine = create_engine(connection_url, module=odbc)
    connection = engine.connect();

    try:
        # execute a simple query
        connection.execute(text("SELECT 1"))
        print("Connection successful!")
    except Exception as e:
        print(f"Connection failed! Error: {e}")
    finally:
        # close the connection
        connection.close()


def copy_audio_files(server, database, username, password, destination_directory, num_files_to_copy):
    # Connection string
    conn_str = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password};TrustServerCertificate=yes;"

    test_db_connection(conn_str)

    # Connect to the SQL Server database
    connection = pyodbc.connect(conn_str)
    cursor = connection.cursor()

    try:
        # Execute the SQL query to get the paths of audio files
        query = """select CC_FILE_LOCATION_UNC from dgv_contact_content
          where co_id in 
            (select co_id from dgv_contact where i_id = 20319 and CO_COMMUNICATIONSTYPE = 17)
              and cc_file_type = 0 and CC_MULTIMEDIA_TYPE = 13 AND -- 0 = PCM, 13 = WAV
              CC_SUB_SEQNR = 0 -- 0 = Target, 1 = other party"""

        cursor.execute(query, (num_files_to_copy,))
        rows = cursor.fetchall()

        # Iterate through the result and copy files to the destination directory
        for row in rows:
            file_path = row.file_path
            # Split the path into segments
            path_segments = file_path.split('/')

            # Join the segments, excluding the first one
            new_path = '/'.join(path_segments[1:])
            file_name = file_path.split("/")[-1]  # Extract the file name from the path
            destination_path = f"{destination_directory}/{file_name}"
            shutil.copy(file_path, destination_path)
            print(f"File copied: {file_name}")

    except pyodbc.Error as e:
        print(f"SQL Server error: {e}")

    finally:
        # Close the database connection
        connection.close()