import pyodbc

try:

    # server_name = 'Local'
    # database_name = 'LIDB'
    # user = 'sa'
    # password = 'Strong@Passw0rd'


    server_name = 'LIDB'
    database_name = 'LIDB'
    user = 'sa'
    password = 'masterkey'


    # Use the FreeTDS configuration file
    connection_string = f"DRIVER=FreeTDS;SERVERNAME={server_name};DATABASE={database_name};UID={user};PWD={password};"

    # Establish the connection
    connection = pyodbc.connect(connection_string)


except pyodbc.Error as e:
    print(f"Error connecting to database: {e}")
    print("SQLSTATE: ", e.args[0])
    print("Message: ", e.args[1])


