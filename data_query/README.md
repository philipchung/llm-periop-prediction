# SQL Data Query Scripts

`data_query` contains a separate poetry environment with dependencies specific to querying data from a SQL Database that houses data from the Electronic Health Record.  This environment is only used to export data from SQL to local files which we can then further process.
## Secrets

Database secrets are stored in file `database_config.py`, which is not commited to this repo.  Provide a file that contains the following:

```python
# database_config.py
server = "<SQL Server Name>"
database = "<Database Name>"
schema = "<Database Schema Name>"
```
