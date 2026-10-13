# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
import socket

import pytest
import pytest_asyncio

from arduino.app_tools.app_launcher.ingress import Ingress


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def echo(reader, writer):
    while data := await reader.read(65536):
        writer.write(data.upper())
        await writer.drain()
    writer.close()


@pytest_asyncio.fixture
async def upstream():
    server = await asyncio.start_server(echo, "127.0.0.1", 0)
    yield server.sockets[0].getsockname()[1]
    server.close()


async def forwarding(lookup, wait_s=2.0):
    public = free_port()
    ingress = Ingress([public], lookup, host="127.0.0.1", wait_s=wait_s)
    await ingress.start()
    return ingress, public


@pytest.mark.asyncio
async def test_forwards_bytes_both_ways(upstream):
    ingress, public = await forwarding(lambda port: upstream)
    reader, writer = await asyncio.open_connection("127.0.0.1", public)
    payload = b"x" * 300_000
    writer.write(payload)
    await writer.drain()
    assert await reader.readexactly(len(payload)) == payload.upper()
    writer.write_eof()
    assert await asyncio.wait_for(reader.read(), 1) == b""
    writer.close()
    ingress.close()


@pytest.mark.asyncio
async def test_drops_the_connection_when_no_app_runs():
    ingress, public = await forwarding(lambda port: None, wait_s=0.2)
    reader, writer = await asyncio.open_connection("127.0.0.1", public)
    assert await asyncio.wait_for(reader.read(), 1) == b""
    writer.close()
    ingress.close()


@pytest.mark.asyncio
async def test_waits_for_an_app_that_appears_later(upstream):
    running = []
    ingress, public = await forwarding(lambda port: upstream if running else None)
    reader, writer = await asyncio.open_connection("127.0.0.1", public)
    writer.write(b"hello")
    await asyncio.sleep(0.2)
    running.append(True)
    assert await asyncio.wait_for(reader.readexactly(5), 1) == b"HELLO"
    writer.close()
    ingress.close()


@pytest.mark.asyncio
async def test_waits_for_an_app_that_is_not_listening_yet():
    port = free_port()
    ingress, public = await forwarding(lambda p: port)
    reader, writer = await asyncio.open_connection("127.0.0.1", public)
    writer.write(b"late")
    await asyncio.sleep(0.2)
    server = await asyncio.start_server(echo, "127.0.0.1", port)
    assert await asyncio.wait_for(reader.readexactly(4), 1) == b"LATE"
    writer.close()
    ingress.close()
    server.close()
