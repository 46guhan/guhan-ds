create database company;
use company;

create table employee(sno int auto_increment not null primary key, sname varchar(200), age int, designaton varchar(250), salary int);
insert into employee(sname,age,designaton,salary) values('hari',19,'applicaton developer',27000);

select * from employee;

select * from employee where salary>25000;
select sname from employee where salary<20000;
select sname,salary from employee where designaton='software devoloper';
select sname,designaton,salary from employee where age=21;

update employee set salary=19000 where designaton='backend support';
delete from employee where age=19;
SET SQL_SAFE_UPDATES = 0;

select * from employee;