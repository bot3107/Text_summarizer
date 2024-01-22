# Text_summarizer
## PostgreSQL 
DATA TYPES in PostgreSQL:
1.	BOOLEAN:
a) TRUE, yes, y, t, 1 they all evaluated as TRUE.
b) FALSE, no, n, f, 0 they all evaluated as FALSE.
2.	CHARACTER
a)	Char (n) – contains character >= n but occupies memory for all n character.  
b) Varchar (n) – contains character >= n but occupies memory only for characters present. 
c)Text – can contain variable length of character best for descriptions.
3.	NUMERIC (for auto increment in MySQL here the alternative is ‘SERIAL’)
a)	Integer: int, smallint are considered inside this. 
b)	Floating number: float(n) here n is the precise number of digits after the decimal point.
c)	Numeric or numeric (p, s): Here the p is number of total digits and s is the number of digits after the decimal point.
4.	TEMPORAL
a)	DATE: only date will be shown when used this.
b)	TIME: only time will be shown.
c)	TIMESTAMP: both DATE and TIME will be shown.
d)	TIMESTAMPZ: both DATE and TIME will be shown based on the time zone provided.
e)	INTERVAL: stores period. 
5.	UUID
This guarantees better uniqueness than serial known as universally unique identifier.  
6.	ARRAY
7.	JSON


CREATE TABLE table_name;
INSERT INTO table_name values (-----------------------------);
SELECT * FROM table_name; 
