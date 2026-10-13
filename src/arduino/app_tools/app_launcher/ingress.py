# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
import logging
from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

CHUNK_BYTES = 64 * 1024
POLL_S = 0.05

Upstream = Callable[[int], int | None]
"""Maps a public port to the internal port of the running app, None while no app runs."""


class Ingress:
    """Forwards every TCP connection on the public ports, byte for byte, to the running app on 127.0.0.1."""

    def __init__(self, ports: Sequence[int], upstream: Upstream, host: str = "0.0.0.0", wait_s: float = 2.0) -> None:
        """Configures the forwarder.

        Args:
            ports (Sequence[int]): The public ports to listen on.
            upstream (Upstream): Resolves the internal port for a public one.
            host (str): The numeric address to listen on: names are never resolved.
            wait_s (float): How long a connection waits for an app that is not running or not listening yet.
        """
        self._ports = tuple(ports)
        self._upstream = upstream
        self._host = host
        self._wait_s = wait_s
        self._servers: list[asyncio.Server] = []

    async def start(self) -> None:
        """Starts listening on every public port."""
        for port in self._ports:
            server = await asyncio.start_server(lambda r, w, p=port: self._forward(p, r, w), self._host, port, limit=CHUNK_BYTES)
            self._servers.append(server)

    def close(self) -> None:
        """Stops listening and drops the open connections."""
        for server in self._servers:
            server.close()
            server.abort_clients()
        self._servers.clear()

    async def _forward(self, port: int, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
        upstream = await self._connect(port)
        if upstream is None:
            client_writer.close()
            return
        upstream_reader, upstream_writer = upstream
        pumps = [asyncio.ensure_future(_pump(client_reader, upstream_writer)), asyncio.ensure_future(_pump(upstream_reader, client_writer))]
        try:
            await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for pump in pumps:
                pump.cancel()
            await asyncio.gather(*pumps, return_exceptions=True)
            client_writer.close()
            upstream_writer.close()

    async def _connect(self, port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter] | None:
        # Waiting covers a browser reload that lands while one app is replaced by the next
        deadline = asyncio.get_running_loop().time() + self._wait_s
        while True:
            internal = self._upstream(port)
            if internal is not None:
                try:
                    return await asyncio.open_connection("127.0.0.1", internal, limit=CHUNK_BYTES)
                except OSError:
                    pass
            if asyncio.get_running_loop().time() >= deadline:
                logger.info(f"Dropped a connection on port {port}: no app is listening")
                return None
            await asyncio.sleep(POLL_S)


async def _pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(CHUNK_BYTES):
            writer.write(data)
            await writer.drain()
    except OSError:
        pass
