--
--
--
--
CREATE TABLE teams (

	id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
	name varchar(150) UNIQUE NOT NULL
);

--
--
--
--
CREATE TABLE accounts (

	id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    team_id INTEGER REFERENCES teams(id) ON DELETE CASCADE NOT NULL,
	first_name VARCHAR NOT NULL,
    last_name VARCHAR NOT NULL, 
    email varchar(150) UNIQUE NOT NULL,
	password_hash varchar NOT NULL
);

--
--
--
--
CREATE TABLE access_rights (

	id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
	name VARCHAR(100) UNIQUE NOT NULL
);

--
--
--
--
CREATE TABLE account_access_rights (

    id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
	account_id INTEGER REFERENCES accounts(id) ON DELETE CASCADE NOT NULL,
	access_right_id INTEGER REFERENCES access_rights(id) ON DELETE CASCADE NOT NULL,
	UNIQUE(account_id, access_right_id)
);

--
--
--
--
CREATE TABLE sessions (

    id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
	account_id INTEGER REFERENCES accounts(id) ON DELETE CASCADE NOT NULL,
	user_agent_hash VARCHAR(64),
	http_only_token VARCHAR(128),
	csrf_token_hash VARCHAR(128),
	last_activity VARCHAR(100),
	last_ip VARCHAR(45),
	browser_name VARCHAR(50),
	browser_version VARCHAR(50),
	os_name VARCHAR(50),
	os_version VARCHAR(50),
	device_type VARCHAR(50),
	create_date DATE NOT NULL,
	update_date DATE
);

--
--
--
--
CREATE TABLE categories (

	id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
	team_id INTEGER REFERENCES teams(id) ON DELETE CASCADE NOT NULL,
    title VARCHAR(150),
    description TEXT
);

--
--
--
--
CREATE TYPE task_status AS ENUM('PREDEFINED', 'TODO', 'IN_PROGRESS', 'DONE', 'ARCHIVED');

--
--
--
--
CREATE TABLE tasks (

	id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    team_id INTEGER REFERENCES teams(id) ON DELETE CASCADE,
    creator_account_id INTEGER REFERENCES accounts(id) ON DELETE CASCADE,
    executor_account_id INTEGER REFERENCES accounts(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES categories(id) ON DELETE CASCADE,
    title VARCHAR NOT NULL,
    description TEXT NOT NULL,
    estimated_hours INTEGER NOT NULL
);

CREATE INDEX idx_tasks_team_id ON tasks (team_id);

--
--
--
--
CREATE TABLE task_events (

	id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    task_id INTEGER REFERENCES tasks(id) ON DELETE CASCADE NOT NULL,
    status task_status NOT NULL,
    time TIMESTAMP NOT NULL
);

CREATE INDEX idx_task_events_task_id_time ON task_events (task_id, time DESC);

--
--
--
--
CREATE TABLE documents (

	id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    account_id INTEGER REFERENCES accounts(id) ON DELETE CASCADE NOT NULL,
	team_id INTEGER REFERENCES teams(id) ON DELETE CASCADE NOT NULL,
    title VARCHAR(250),
    description TEXT,
    file_name TEXT,
	mime VARCHAR
);