"""Source: the cache in front of every fetch, and the contract the base holds a client to."""

from unittest.mock import patch

import httpx
import pytest

from fopol.base.dataset import Dataset
from fopol.base.source import Kind, UnsupportedKind, cached_get, cached_get_many
from fopol.data.result import Result
from fopol.data.team import TeamStat
from fopol.sources.fpl import FPLArchiveSource


@patch("fopol.base.source.httpx.get")
def test_cached_get_fetches_once_then_reads_from_disk(get, tmp_path):
    """The second call for a URL never leaves the machine.

    That is what makes a load reproducible.
    """
    get.return_value.status_code = 200
    get.return_value.text = "a,b\n1,2\n"

    first = cached_get("https://example.test/x.csv", cache_dir=tmp_path)
    second = cached_get("https://example.test/x.csv", cache_dir=tmp_path)

    assert first == second == "a,b\n1,2\n"
    get.assert_called_once()
    assert get.call_args.args == ("https://example.test/x.csv",)
    assert get.call_args.kwargs["follow_redirects"] is True


@patch("fopol.base.source.httpx.get")
def test_cached_get_remembers_a_404_as_none(get, tmp_path):
    """A missing per-player file is a fact, not a retry: the 404 is cached and returns ``None``."""
    get.return_value.status_code = 404

    assert cached_get("https://example.test/missing.csv", cache_dir=tmp_path) is None
    assert cached_get("https://example.test/missing.csv", cache_dir=tmp_path) is None
    get.assert_called_once()


@patch("fopol.base.source.httpx.get")
def test_cached_get_raises_on_other_http_errors(get, tmp_path):
    """Anything but 200 or 404 is an error the caller must see.

    Nothing is written to the cache on the way out.
    """
    get.return_value.status_code = 503
    request = httpx.Request("GET", "https://example.test/down.csv")
    get.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
        "503", request=request, response=httpx.Response(503, request=request)
    )

    with pytest.raises(httpx.HTTPStatusError):
        cached_get("https://example.test/down.csv", cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


@patch("fopol.base.source.httpx.get")
def test_cached_get_refresh_bypasses_the_cache(get, tmp_path):
    """``refresh=True`` refetches and overwrites, for feeds that change under the same URL."""
    get.return_value.status_code = 200
    get.return_value.text = "v1"
    cached_get("https://example.test/live.json", cache_dir=tmp_path)
    get.return_value.text = "v2"

    assert cached_get("https://example.test/live.json", cache_dir=tmp_path, refresh=True) == "v2"
    assert cached_get("https://example.test/live.json", cache_dir=tmp_path) == "v2"
    assert get.call_count == 2


@patch("fopol.base.source.cached_get")
def test_cached_get_many_preserves_order(cached):
    """Concurrent fetches come back in the order asked, whatever order they finish in."""
    cached.side_effect = lambda url, **_: url.rsplit("/", 1)[-1]

    assert cached_get_many(["u/a", "u/b", "u/c"], workers=3) == ["a", "b", "c"]
    assert cached.call_count == 3


@patch.object(FPLArchiveSource, "provides", {Kind.RESULTS: Result})
def test_get_refuses_a_kind_the_source_does_not_declare():
    """A source's ``provides`` is its promise.

    Asking for anything else is a typed error, not a KeyError.
    """
    with pytest.raises(UnsupportedKind, match="appearances"):
        FPLArchiveSource().appearances("2024-25")


@patch.object(FPLArchiveSource, "_fetch")
def test_get_rejects_a_fetch_that_breaks_its_declaration(fetch, season):
    """``_fetch`` returning a record type other than the declared one is caught at the base."""
    fetch.return_value = Dataset([], record_type=TeamStat)

    with pytest.raises(TypeError, match="TeamStat"):
        FPLArchiveSource().results(season)


@patch.object(FPLArchiveSource, "_fetch")
def test_get_memoises_per_kind_and_season(fetch, season):
    """One fetch per (kind, season) for the life of the client."""
    fetch.return_value = Dataset([], record_type=Result)
    source = FPLArchiveSource()

    assert source.results(season) is source.results(season)
    fetch.assert_called_once_with(Kind.RESULTS, season)


@patch.object(FPLArchiveSource, "_fetch")
def test_load_concatenates_seasons_under_the_source_name(fetch):
    """``load`` walks the declared seasons in order.

    The union is tagged with the source's provenance, not the season's.
    """
    fetch.return_value = Dataset([], record_type=Result)
    source = FPLArchiveSource(first_season="2022-23", last_season="2023-24")

    data = source.load("results")

    assert data.record_type is Result
    assert data.source == "fpl-archive"
    assert [c.args[1] for c in fetch.call_args_list] == ["2022-23", "2023-24"]
