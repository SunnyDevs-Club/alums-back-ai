from flask import request

from app.models import Group, User
from app.groups import bp


@bp.get('/<int:group_id>')
def get_single_group(group_id):
    """
    Get single user by its user_id
    """
    group = Group.get(group_id)
    if not group:
        return {"message": "User not found"}, 404

    return group.to_dict(), 200


@bp.delete('/<int:group_id>')
def delete_group(group_id):
    """
    Delete user
    """
    group = Group.get(group_id)
    if not group:
        return {"message": "User not found"}, 404

    group.delete()

    return {"message": "User deleted successfully"}, 200


@bp.get('/')
def get_all_groups():
    """
    Get All users. filter with query parameters: 'role'
    """
    groups = Group.get_all()
    groups_data = [group.to_dict() for group in groups]

    return {
        "total": len(groups_data),
        "data": groups_data
    }


@bp.post('/')
def create_group():
    """
    Create Group. Expecting data:
    {
        requested_from: int - user id.
        data: {
            group_name: str - group name
        }
    }
    """
    data = request.get_json()

    admin = User.get(data['requested_from'])

    if admin.role != 'admin':
        return {
            "message": "Only admin can create new users"
        }, 400

    new_group = Group(group_name=data['data']['group_name'])
    new_group.add()

    return {
        "message": "Created successfuly"
    }, 201
