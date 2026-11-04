"""Minimal MQTT 3.1.1 broker for the local lab fixture."""
import asyncio
import logging
import struct

logger = logging.getLogger(__name__)


def _remaining(n: int) -> bytes:
    out = bytearray()
    while True:
        digit = n % 128
        n //= 128
        if n:
            digit |= 0x80
        out.append(digit)
        if not n:
            return bytes(out)


class LabMqttBroker:
    def __init__(self, host: str = "127.0.0.1", port: int = 1883):
        self.host = host
        self.port = port
        self.messages: list[tuple[str, bytes]] = []
        self._server: asyncio.base_events.Server | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, self.port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _read_packet(self, reader: asyncio.StreamReader):
        first = await reader.readexactly(1)
        remaining = 0
        multiplier = 1
        while True:
            digit = (await reader.readexactly(1))[0]
            remaining += (digit & 0x7F) * multiplier
            if not digit & 0x80:
                break
            multiplier *= 128
        body = await reader.readexactly(remaining) if remaining else b""
        return first[0] >> 4, first[0] & 0x0F, body

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                packet_type, flags, body = await self._read_packet(reader)
                logger.info("mqtt packet type=%s bytes=%s", packet_type, len(body))
                if packet_type == 1:  # CONNECT
                    writer.write(bytes([0x20, 0x02, 0x00, 0x00]))
                elif packet_type == 3:  # PUBLISH
                    self._accept_publish(flags, body)
                    if flags & 0x06:
                        topic_len = struct.unpack("!H", body[:2])[0]
                        packet_id = body[2 + topic_len:4 + topic_len]
                        writer.write(bytes([0x40, 0x02]) + packet_id)
                elif packet_type == 8:  # SUBSCRIBE
                    packet_id = body[:2]
                    writer.write(bytes([0x90, 0x03]) + packet_id + bytes([0x00]))
                elif packet_type == 12:  # PINGREQ
                    writer.write(bytes([0xD0, 0x00]))
                elif packet_type == 14:  # DISCONNECT
                    break
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            pass
        except Exception:
            logger.exception("lab mqtt client failed")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    def _accept_publish(self, flags: int, body: bytes) -> None:
        topic_len = struct.unpack("!H", body[:2])[0]
        topic = body[2:2 + topic_len].decode("utf-8", errors="replace")
        start = 2 + topic_len + (2 if flags & 0x06 else 0)
        self.messages.append((topic, body[start:]))
