"""Unit tests for the Content-Length header parser."""

from __future__ import annotations

from baca.rpc_client import RpcClient


def test_shouldParseContentLengthWhenStandardHeader() -> None:
    # given
    header = "Content-Length: 42"

    # when
    result = RpcClient._parse_content_length(header)

    # then
    assert result == 42


def test_shouldParseContentLengthWhenMixedCase() -> None:
    # given
    header = "content-length: 7"

    # when
    result = RpcClient._parse_content_length(header)

    # then
    assert result == 7


def test_shouldReturnNoneWhenHeaderMalformed() -> None:
    # given
    header = "Content-Length: not-a-number"

    # when
    result = RpcClient._parse_content_length(header)

    # then
    assert result is None


def test_shouldReturnNoneWhenHeaderMissing() -> None:
    # given
    header = "X-Other: 1"

    # when
    result = RpcClient._parse_content_length(header)

    # then
    assert result is None
