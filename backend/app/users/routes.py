from flask import request

from app.models import User
from app.users import bp


@bp.get('/<int:user_id>')
def get_single_user(user_id):
    """
    Get single user by its user_id
    """
    user = User.get(user_id)
    if not user:
        return {"message": "User not found"}, 404

    return user.to_dict(), 200


@bp.delete('/<int:user_id>')
def delete_user(user_id):
    """
    Delete user
    """
    user = User.get(user_id)
    if not user:
        return {"message": "User not found"}, 404

    user.delete()

    return {"message": "User deleted successfully"}, 200


@bp.get('/')
def get_all_users():
    """
    Get All users. filter with query parameters: 'role'
    """
    users = User.get_all()
    users_data = [user.to_dict() for user in users]

    return {
        "total": len(users_data),
        "data": users_data
    }


@bp.post('/')
def create_user():
    """
    Create User. Expecting data:
    {
        requested_from: int - user id.
        data: {
            email: str,
            password: str,
            first_name: str,
            last_name: str,
            role: str - worker | admin,
            group_id: int
        }
    }
    """
    data = request.get_json()

    admin = User.get(data['requested_from'])

    if admin.role != 'admin':
        return {
            "message": "Only admin can create new users"
        }, 400

    user_data = {
        **data['data']
    }

    new_user = User(created_by=admin.user_id, **user_data)
    new_user.add()

    return {
        "message": "Created successfuly"
    }, 201
