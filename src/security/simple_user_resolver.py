from vanna.core.user import RequestContext, User, UserResolver


class SimpleUserResolver(UserResolver):
    async def resolve_user(self, request_context: RequestContext) -> User:
        user_email = request_context.get_cookie("vanna_email") or "admin@example.com"
        group = "admin" if "admin" in user_email else "user"
        return User(id=user_email, email=user_email, group_memberships=[group])
