## PostgreSQL

Data from prompts and LLM generations are stored in PostgreSQL.  Make sure that `postgresql` and `libpq-dev` are installed.

To configure user accounts and set passwords, you may need to set PostgreSQL's `pg_hba.conf` to `trust` for `local`.
(https://stackoverflow.com/questions/18664074/getting-error-peer-authentication-failed-for-user-postgres-when-trying-to-ge)
Make sure to `sudo service postgresql restart` after.


Create Database
```sh
# Superuser Access to PostgreSQL--default super username=postgres
sudo -u postgres psql

# Create Database
CREATE DATABASE postgresdb 
WITH OWNER = postgres
    ENCODING ='UTF8'
    LC_COLLATE = 'C'
    LC_CTYPE = 'en_US.UTF-8' 
    CONNECTION LIMIT = -1
    TEMPLATE template0;

# Give Default User `postgres` a password
ALTER USER postgres with encrypted PASSWORD 'postgres';

# Give Default User all privileges on newly created database
GRANT all privileges on DATABASE postgresdb to postgres;

# Create new user
sudo -u postgres createuser <username>
# Create Database
sudo -u postgres createdb <dbname>
```

We backup the database maually using `pg_dump`

```sh
sudo pg_dump -U postgres -W -F t postgresdb > /path/to/backup/postgresdb/backup.tar
```

To restore the database.  The `-C` creates the database before loading data into it.
You may alternatively create the database first, and then call the below command without `-C`.
```sh
pg_restore -d postgresdb -U postgres -C "/path/to/backup/postgresdb/backup.tar"
```