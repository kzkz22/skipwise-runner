"""The video's language for captions comes from the audio track YouTube marks original, not from yt-dlp's
top-level `language`: with auto-dubbed tracks that was a dub's (an English video came back with Arabic captions)."""
from dwyt_runner.captions import audio_language, choose_track

FMT = [{"format_id": "json3", "ext": "json3", "url": "https://example.invalid/x"}]


def _dubbed_info():
    """The shape yt-dlp gave for uBuj87HsWt0 (English talk, 14 auto-dubs, manual English subtitles)."""
    formats = [{"format_id": "233-0", "acodec": "mp4a", "language": "ar", "format_note": "العربية - dubbed-auto"},
               {"format_id": "139-0", "acodec": "mp4a", "language": "ar", "format_note": "Arabic, low"},
               {"format_id": "140-14", "acodec": "mp4a", "language": "en-US", "language_preference": 10,
                "format_note": "English (US) original (default), medium"},
               {"format_id": "251-12", "acodec": "opus", "language": "ar", "format_note": "Arabic, medium"},
               {"format_id": "137", "acodec": "none", "language": None}]
    auto = {f"{code}-orig": FMT for code in ("en", "iw", "ja", "ko", "ar")}
    auto.update({"ar": FMT, "en": FMT, "hu-en": FMT})
    return {"language": "ar", "formats": formats, "subtitles": {"en": FMT}, "automatic_captions": auto}


def test_the_original_audio_track_decides_not_the_dub():
    info = _dubbed_info()
    assert audio_language(info) == "en-US"
    assert choose_track(info)[:2] == ("manual", "en")
    info["subtitles"] = {}
    assert choose_track(info)[:2] == ("auto", "en-orig")


def test_the_asked_language_still_wins_and_no_marked_track_keeps_the_old_way():
    info = _dubbed_info()
    assert choose_track(info, "ja")[:2] == ("auto", "ja-orig")
    plain = {"language": "de", "formats": [{"acodec": "opus", "language": "de"}], "subtitles": {},
             "automatic_captions": {"de-orig": FMT}}
    assert audio_language(plain) is None and choose_track(plain)[:2] == ("auto", "de-orig")
    # a format note saying original counts even without the preference value
    noted = {"formats": [{"acodec": "opus", "language": "fr", "format_note": "French original"}],
             "subtitles": {}, "automatic_captions": {"fr-orig": FMT, "en-orig": FMT}}
    assert choose_track(noted)[:2] == ("auto", "fr-orig")
