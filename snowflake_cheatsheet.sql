--> Add timestamp to a table
select * 
        , convert_timezone('America/New_York', current_timestamp()) as tstamp
        , concat(month(tstamp), '/', day(tstamp), '/', year(tstamp), ' ', hour(tstamp), ':', minute(tstamp)) as created_time
        from some_table

--> Modify snowflake date-time to a format that Google Data Studio does not mess up
--> to_char converts to character and to_date converts the snowflake time format to a date
to_char(to_date(some_date)) as some_date