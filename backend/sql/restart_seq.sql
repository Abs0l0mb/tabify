SELECT pg_get_serial_sequence('teams', 'id');
ALTER SEQUENCE teams_id_seq RESTART WITH 10;