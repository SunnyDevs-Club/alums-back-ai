import datetime

from flask import request

import os

from config import Config
from app.parcels import bp
from app.models import Parcel, CropType, User, Task


@bp.post('/crops')
def add_crop_type():
    """
    Data Format:
    {
        requested_from: int,
        data: {
            crop_name: str,
            crop_id: int,
        }
    }
    Returns:

    """
    data = request.get_json()

    user = User.get(data['requested_from'])

    if user.role != 'admin':
        return {
            'message': "Only admin can add new crops"
        }, 400

    new_crop_type = CropType(**data['data'])
    new_crop_type.add()

    return {
        "message": "Created successfully"
    }, 201


@bp.get('/crops')
def get_all_crop_types():
    crops = CropType.get_all()
    return {
        "data": [crop.to_dict() for crop in crops],
        "total": len(crops)
    }, 200


@bp.get('/crops/<int:crop_id>')
def get_crop_by_id(crop_id):
    return CropType.get(crop_id).to_dict(), 200


@bp.get('/')
def get_parcels():
    """
    Body: region, district. Both must be present
    Returns:

    """
    data = request.get_json()
    if 'region' in data and 'district' in data:
        parcels = Parcel.get_all(region=data['region'], district=data['district'])

        return {
            'data': [parcel.to_dict() for parcel in parcels],
            'total': len(parcels)
        }, 200


@bp.post('/<int:parcel_id>')
def update_parcel(parcel_id):
    """
    Data Format:
    {
        requested_from: int,
        data: {
            image: filename format <user_id>_<task_id>_<crop_id>.<ext>,
        }
    }
    Args:
        parcel_id:
    """

    parcel = Parcel.get(parcel_id)

    if 'file' not in request.files:
        return {"message": "No file part"}, 400

    file = request.files['file']
    if file.filename == '':
        return {"message": "No selected file"}, 400

    if file:
        user_id, task_id, crop_id_with_ext = file.filename.split('_')
        crop_id, ext = crop_id_with_ext.split('.')

        current_user = User.get(user_id)
        current_task = Task.get(task_id)

        if current_user.role != 'worker':
            return {"message": "Error! Incorrect User"}, 400

        filepath = os.path.join(Config.UPLOAD_FOLDER, f"{datetime.datetime.now()}_{file.filename}")
        file.save(filepath)

        parcel.last_operator_id = user_id
        parcel.last_photo_path = filepath
        parcel.last_operator_crop = crop_id
        parcel.current_task_is_checked = True

        parcel.update()

        if all(parcel.current_task_is_checked for parcel in current_task.parcels):
            current_task.status = 'completed'
            current_task.update()

        return {"message": "Parcel updated"}, 204
