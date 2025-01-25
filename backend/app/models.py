import datetime
import json

from geoalchemy2 import types as geo_types
import geoalchemy2.functions as func
from werkzeug.security import generate_password_hash, check_password_hash

from app.extensions import db


class User(db.Model):
    __tablename__ = 'user'  # The name of the table in your database
    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.String, unique=True, nullable=False)
    password_hash = db.Column(db.String, nullable=False)

    first_name = db.Column(db.String, nullable=False)
    last_name = db.Column(db.String, nullable=False)
    role = db.Column(db.Enum('admin', 'worker', name='user_role'), nullable=False, default='worker')
    group_id = db.Column(db.Integer, db.ForeignKey('group.group_id'))

    created_on = db.Column(db.DateTime, default=db.func.current_timestamp())
    created_by = db.Column(db.Integer, db.ForeignKey('user.user_id'))
    modified_on = db.Column(db.DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())
    modified_by = db.Column(db.Integer, db.ForeignKey('user.user_id'))

    created_by_user = db.relationship("User", foreign_keys=[created_by])
    modified_by_user = db.relationship("User", foreign_keys=[modified_by])

    @property
    def password(self):
        raise AttributeError('password: write-only field')

    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @classmethod
    def get(cls, user_id) -> 'User':
        return db.get_or_404(cls, user_id)

    @classmethod
    def get_all(cls) -> list['User']:
        return db.session.execute(db.select(User).order_by(User.last_name)).scalars()

    def update(self):
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e
        finally:
            db.session.close()

    def to_dict(self):
        return {
            'user_id': self.user_id,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'role': self.role,
            'group_id': self.group_id
        }

    def delete(self):
        db.session.delete(self)
        db.session.commit()

    def add(self):
        db.session.add(self)
        db.session.commit()


class Group(db.Model):
    __tablename__ = 'group'
    group_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    group_name = db.Column(db.String, nullable=False)

    tasks = db.relationship('Task', back_populates="group")

    @classmethod
    def get(cls, group_id) -> 'Group':
        return db.get_or_404(cls, group_id)

    @classmethod
    def get_all(cls) -> list['Group']:
        return db.session.execute(db.select(Group).order_by(Group.group_name)).scalars()

    @staticmethod
    def update(self):
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e
        finally:
            db.session.close()

    def to_dict(self):
        return {
            'group_id': self.group_id,
            'group_name': self.group_name
        }

    def delete(self):
        db.session.delete(self)
        db.session.commit()

    def add(self):
        db.session.add(self)
        db.session.commit()


class Task(db.Model):
    __tablename__ = 'task'

    task_id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    name = db.Column(db.String, nullable=False)
    description = db.Column(db.Text, nullable=False)

    admin_id = db.Column(db.BigInteger, db.ForeignKey('user.user_id'), nullable=False)
    worker_id = db.Column(db.BigInteger, db.ForeignKey('user.user_id'), nullable=False)
    group_id = db.Column(db.Integer, db.ForeignKey('group.group_id'), nullable=False)

    assignment_date = db.Column(db.Date, nullable=False, default=datetime.date.today())
    deadline_date = db.Column(db.Date, nullable=False)
    status = db.Column(db.Enum('in_progress', 'completed', 'overdue', name='task_status'),
                       default='in_progress')

    admin_user = db.relationship("User", foreign_keys=[admin_id])
    worker_user = db.relationship("User", foreign_keys=[worker_id])
    group = db.relationship("Group", back_populates="tasks")

    parcels = db.relationship("Parcel", back_populates="current_task_obj")

    @classmethod
    def get(cls, task_id) -> 'Task':
        return db.get_or_404(cls, task_id)

    @classmethod
    def get_all(cls) -> list['Task']:
        return db.session.execute(db.select(Task).order_by(Task.name)).scalars()

    @classmethod
    def get_for_admin(cls, admin_id) -> list['Task']:
        return db.session.execute(db.select(Task).where(Task.admin_id == admin_id).order_by(Task.deadline_date)).scalars()

    @classmethod
    def get_for_worker(cls, worker_id) -> list['Task']:
        return db.session.execute(db.select(Task).where(Task.worker_id == worker_id).order_by(Task.deadline_date)).scalars()

    @staticmethod
    def update():
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e
        finally:
            db.session.close()

    def to_dict(self):
        return {
            'task_id': self.task_id,
            'name': self.name,
            'description': self.description,
            'assignment_date': self.assignment_date,
            'deadline_date': self.deadline_date,
            'status': self.status,
            'worker': self.worker_user.to_dict(),
            'admin': self.admin_user.to_dict(),
            'group': self.group.to_dict(),
            'parcels': [parcel.to_dict() for parcel in self.parcels]
        }

    def delete(self):
        db.session.delete(self)
        db.session.commit()

    def add(self):
        db.session.add(self)
        db.session.commit()


class CropType(db.Model):
    __tablename__ = 'crop_type'

    crop_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False, unique=True)

    @classmethod
    def get(cls, crop_id) -> 'CropType':
        return db.get_or_404(cls, crop_id)

    @classmethod
    def id_to_name(cls, crop_id):
        return cls.get(crop_id).name

    @classmethod
    def get_all(cls) -> list['CropType']:
        return db.session.execute(db.select(CropType).order_by(CropType.name)).scalars()

    @staticmethod
    def update():
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e
        finally:
            db.session.close()

    def to_dict(self):
        return {
            'crop_id': self.crop_id,
            'crop_name': self.name
        }

    def delete(self):
        db.session.delete(self)
        db.session.commit()

    def add(self):
        db.session.add(self)
        db.session.commit()


class Parcel(db.Model):
    __tablename__ = 'parcel'

    parcel_id = db.Column(db.BigInteger, primary_key=True)
    parcel_geom = db.Column(geo_types.Geometry('Polygon', 4326), nullable=False)

    owner_name = db.Column(db.String, nullable=False)
    mfy = db.Column(db.String, nullable=False)
    district = db.Column(db.String, nullable=False)
    region = db.Column(db.String, nullable=False)

    kontur_number = db.Column(db.Float, nullable=False)

    farmer_crop = db.Column(db.SmallInteger, db.ForeignKey('crop_type.crop_id'), nullable=False)
    classified_crop = db.Column(db.SmallInteger, db.ForeignKey('crop_type.crop_id'), nullable=True)

    current_task = db.Column(db.Integer, db.ForeignKey('task.task_id'), nullable=True)

    last_operator_crop = db.Column(db.SmallInteger, db.ForeignKey('crop_type.crop_id'))
    last_operator_id = db.Column(db.BigInteger, db.ForeignKey('user.user_id'))
    last_photo_path = db.Column(db.String)
    last_checked_on = db.Column(db.DateTime)

    farmer_crop_type = db.relationship('CropType', foreign_keys=[farmer_crop])
    classified_crop_type = db.relationship('CropType', foreign_keys=[classified_crop])
    last_operator_crop_type = db.relationship('CropType', foreign_keys=[last_operator_crop])
    last_operator = db.relationship('User', foreign_keys=[last_operator_id])

    current_task_is_checked = db.Column(db.Boolean, default=False)

    current_task_obj = db.relationship('Task', back_populates="parcels")

    @classmethod
    def get(cls, parcel_id) -> 'Parcel':
        return db.get_or_404(cls, parcel_id)

    @classmethod
    def get_all(cls, region, district) -> list['Parcel']:
        return db.session.execute(db.select(Parcel).where(Parcel.region == region, Parcel.district == district).order_by(Parcel.mfy)).scalars()

    @staticmethod
    def update():
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e
        finally:
            db.session.close()

    def to_dict(self):
        geom_json = db.session.scalar(func.ST_AsGeoJSON(self.geom))
        return {
            'parcel_id': self.parcel_id,
            'parcel_geom': geom_json,
            'owner_name': self.owner_name,
            'mfy': self.mfy,
            'district': self.district,
            'region': self.region,
            'kontur_number': self.kontur_number,
            'cadastre_number': self.cadastre_number,
            'last_checked_on': self.last_checked_on.isoformat() if self.last_checked_on else None,
            'farmer_crop_type': self.farmer_crop_type.to_dict() if self.farmer_crop_type else None,
            'classified_crop_type': self.classified_crop_type.to_dict() if self.classified_crop_type else None,
            'last_operator_crop_type': self.last_operator_crop_type.to_dict() if self.last_operator_crop_type else None,
            'last_operator': self.last_operator.to_dict() if self.last_operator else None,
            'current_task': self.current_task if self.current_task else None,
            'current_task_is_checked': self.current_task_is_checked
        }

    def delete(self):
        db.session.delete(self)
        db.session.commit()

    def add(self):
        db.session.add(self)
        db.session.commit()