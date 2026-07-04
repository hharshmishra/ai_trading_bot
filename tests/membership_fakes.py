"""Shared fakes for the membership test files (not collected — no test_ prefix)."""
from __future__ import annotations

from types import SimpleNamespace


class FakeMessage:
    def __init__(self):
        self.replies = []

    async def reply_text(self, text, **kw):
        self.replies.append({"text": text, **kw})
        return SimpleNamespace(message_id=len(self.replies))


class FakeBot:
    def __init__(self, username="PayBotTest"):
        self.username = username
        self.sent, self.invites, self.banned, self.unbanned = [], [], [], []
        self.raise_for_uids = set()          # send_message failure injection

    async def send_message(self, chat_id, text, **kw):
        if chat_id in self.raise_for_uids:
            raise RuntimeError("blocked by user")
        self.sent.append({"chat_id": chat_id, "text": text, **kw})
        return SimpleNamespace(message_id=len(self.sent))

    async def create_chat_invite_link(self, chat_id, **kw):
        self.invites.append({"chat_id": chat_id, **kw})
        return SimpleNamespace(invite_link=f"https://t.me/+inv{len(self.invites)}")

    async def ban_chat_member(self, chat_id, user_id):
        self.banned.append((chat_id, user_id))

    async def unban_chat_member(self, chat_id, user_id):
        self.unbanned.append((chat_id, user_id))

    async def get_me(self):
        return SimpleNamespace(username=self.username)


class FakeCQ:
    def __init__(self, data, uid=101, username="u"):
        self.data = data
        self.from_user = SimpleNamespace(id=uid, username=username)
        self.message = FakeMessage()
        self.answers = []

    async def answer(self, text="", show_alert=False):
        self.answers.append(text)


class FakeJoinRequest:
    def __init__(self, uid):
        self.from_user = SimpleNamespace(id=uid)
        self.approved = self.declined = False

    async def approve(self):
        self.approved = True

    async def decline(self):
        self.declined = True


def mk_ctx(bd, bot, args=None):
    return SimpleNamespace(application=SimpleNamespace(bot_data=bd), bot=bot,
                           args=args or [])


def mk_cmd_update(uid=101, username="u"):
    return SimpleNamespace(effective_user=SimpleNamespace(id=uid, username=username),
                           message=FakeMessage())


class FakeResp:
    def __init__(self, payload, code=200):
        self._p, self.status_code = payload, code

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class FakeRzpHttp:
    """Scripted Razorpay REST fake: created links + a status you can flip."""
    def __init__(self):
        self.statuses = {}                   # link_id -> status
        self.posts, self.gets = [], []

    def post(self, url, auth=None, json=None, timeout=None):
        self.posts.append({"url": url, "auth": auth, "json": json})
        lid = f"plink_{len(self.posts)}"
        self.statuses[lid] = "created"
        return FakeResp({"id": lid, "short_url": f"https://rzp.io/{lid}"})

    def get(self, url, auth=None, timeout=None, **kw):
        self.gets.append(url)
        lid = url.rsplit("/", 1)[-1]
        return FakeResp({"status": self.statuses.get(lid, "created")})


class FakeTronHttp:
    """Scripted TronGrid fake: preload .transfers (raw API shape)."""
    def __init__(self, wallet):
        self.wallet = wallet
        self.transfers = []                  # [{'to','value','transaction_id','block_timestamp'}]

    def get(self, url, params=None, headers=None, timeout=None):
        return FakeResp({"data": self.transfers})
