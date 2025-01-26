import datetime

from flask import request

from app.tasks import bp
from app.models import Task, User, Parcel, db


@bp.post("/")
def create_task():
    """
    Methods: post
    Body: {
        requested_from: int
        data: {
            name: str,
            description: str,
            worker_id: int,
            group_id: int,
            deadline_date: ISO Format,
            parcels: [
                parcel_ids: int
            ]
        }
    }
    Returns:

    """

    body = request.get_json()
    data = body['data']

    current_user = User.get(body['requested_from'])

    if current_user.role != 'admin':
        return {
            "message": 'You do not have permission to create this task'
        }, 400

    parsed_date = datetime.datetime.fromisoformat(data['deadline_date'].replace("Z", "+00:00")).date()
    new_task = Task(
        name=data['name'],
        description=data['description'],
        admin_id=body['requested_from'],
        worker_id=data['worker_id'],
        group_id=data['group_id'],
        deadline_date=parsed_date
    )

    db.session.add(new_task)

    for parcel_id in data['parcels']:
        parcel = Parcel.get(parcel_id)
        parcel.current_task = new_task.task_id
        parcel.current_task_is_checked = False
    
    db.session.commit()
    

    return {
        "message": "Successfully created new task"
    }, 201


@bp.get("/")
def get_tasks():
    """
    body: {
        requested_from: int
    }
    """

    current_user = User.get(request.json['requested_from'])
    if current_user.role == 'admin':
        tasks = Task.get_for_admin(admin_id=current_user.user_id)
    else:
        tasks = Task.get_for_worker(worker_id=current_user.user_id)

    return {
        "total": len(tasks),
        "data": [task.to_dict() for task in tasks]
    }
