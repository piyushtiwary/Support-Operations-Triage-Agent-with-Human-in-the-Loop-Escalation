from contextlib import asynccontextmanager

import httpx
import pytest

from main import create_app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def running_client():
    @asynccontextmanager
    async def start(path, **options):
        app = create_app(path, **options)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                yield client

    return start


@pytest.fixture
def database(tmp_path):
    return tmp_path / "tickets.sqlite3"


@pytest.fixture
async def client(database, running_client):
    async with running_client(database) as client:
        yield client
