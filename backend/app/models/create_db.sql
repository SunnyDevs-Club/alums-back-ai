CREATE TYPE user_role AS ENUM ('admin', 'worker');

CREATE TABLE "user" (
    user_id bigint primary key ,
    email varchar unique not null ,
    password_hash varchar not null ,

    first_name varchar not null ,
    last_name varchar not null ,
    middle_name varchar null ,

    role user_role not null default 'worker',

    created_on timestamp default CURRENT_TIMESTAMP,
    created_by bigint null ,
    modified_on timestamp default CURRENT_TIMESTAMP,
    modified_by bigint null ,

    constraint fk_user_creator foreign key(created_by) references "user"(user_id),
    constraint fk_user_modifier foreign key(modified_by) references "user"(user_id)
);

create type task_status as enum ('in_progress', 'completed', 'overdue', 'completed_late');

create table "group" (
    group_id serial primary key ,
    group_name varchar not null ,
    head_id bigint not null ,

    constraint fk_group_head foreign key (head_id) references "user"(user_id)
);

CREATE TABLE "task" (
    task_id serial primary key ,
    name varchar not null ,
    description text not null ,

    admin_id bigint not null ,
    worker_id bigint not null ,
    group_id int not null ,

    assignment_date date not null ,
    deadline_date date not null ,
    status task_status default 'in_progress',

    constraint fk_admin_task foreign key (admin_id) references "user"(user_id),
    constraint fk_worker_task foreign key (worker_id) references "user"(user_id),
    constraint fk_group_task foreign key (group_id) references "group"(group_id),

    constraint chk_deadline_date check ( deadline_date > task.assignment_date )
);

create table "crop_type" (
    crop_id smallint primary key ,
    crop_name varchar not null unique
);

CREATE TABLE "parcel" (
    parcel_id bigint primary key ,
    parcel_geom geometry(Polygon, 4326) not null ,

    owner_name varchar not null ,
    mfy varchar not null ,
    district varchar not null ,
    region varchar not null ,

    kontur_number real not null ,
    cadastre_number real not null ,

    farmer_crop smallint not null ,
    classified_crop smallint not null ,

    last_operator_crop smallint null ,
    last_operator_id bigint null ,
    last_photo_path varchar null ,
    last_checked_on timestamp null ,

    current_task int null ,

    CONSTRAINT fk_user_parcel FOREIGN KEY (last_operator_id)
        REFERENCES "user"(user_id),

    constraint fk_farmer_crop foreign key (farmer_crop) references "crop_type"(crop_id),
    constraint fk_ai_crop foreign key (classified_crop) references "crop_type"(crop_id),
    constraint fk_operator_crop foreign key (last_operator_crop) references "crop_type"(crop_id),

    constraint fk_task_parcel foreign key (current_task) references "task"(task_id)
);

COMMENT ON COLUMN parcel.mfy IS 'Mahalla Fuqarolar Yigini';


-- Indexes

create index idx_region on parcel(region);
create index idx_district on parcel(district);
create index idx_geometry on parcel using gist(parcel_geom);


-- Triggers

CREATE OR REPLACE FUNCTION validate_geometry()
    RETURNS TRIGGER AS $$
BEGIN
    IF NOT ST_IsValid(NEW.parcel_geom) THEN
        -- Attempt to fix the geometry
        NEW.parcel_geom := ST_MakeValid(NEW.parcel_geom);

        IF NOT ST_IsValid(NEW.parcel_geom) THEN
            RAISE EXCEPTION 'Invalid geometry cannot be fixed';
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER validate_geometry_trigger
    BEFORE INSERT OR UPDATE ON parcel
    FOR EACH ROW
EXECUTE FUNCTION validate_geometry();
