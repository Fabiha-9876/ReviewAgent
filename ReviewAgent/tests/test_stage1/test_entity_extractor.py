"""Tests for the entity extractor — regex patterns and merging."""

import pytest
import asyncio
from src.common.schemas import ExtractedEntities
from src.stage1.entity_extractor import EntityExtractor, DEVICE_PATTERNS, OS_PATTERNS, VERSION_PATTERNS
import re


class TestRegexPatterns:
    """Test regex pattern matching without any LLM dependency."""

    def test_iphone_detection(self):
        text = "Not working on my iPhone 15 Pro"
        matches = []
        for pattern in DEVICE_PATTERNS:
            matches.extend(re.findall(pattern, text, re.IGNORECASE))
        assert any("iPhone 15 Pro" in m for m in matches)

    def test_samsung_detection(self):
        text = "Crashes on Samsung Galaxy S24"
        matches = []
        for pattern in DEVICE_PATTERNS:
            matches.extend(re.findall(pattern, text, re.IGNORECASE))
        assert any("Samsung Galaxy S24" in m for m in matches)

    def test_pixel_detection(self):
        text = "Works fine on my Google Pixel 8"
        matches = []
        for pattern in DEVICE_PATTERNS:
            matches.extend(re.findall(pattern, text, re.IGNORECASE))
        assert any("Pixel 8" in m for m in matches)

    def test_ipad_detection(self):
        text = "Layout broken on iPad Pro"
        matches = []
        for pattern in DEVICE_PATTERNS:
            matches.extend(re.findall(pattern, text, re.IGNORECASE))
        assert any("iPad Pro" in m for m in matches)

    def test_android_version(self):
        text = "Running Android 15 on my phone"
        matches = []
        for pattern in OS_PATTERNS:
            matches.extend(re.findall(pattern, text, re.IGNORECASE))
        assert any("Android 15" in m for m in matches)

    def test_ios_version(self):
        text = "Updated to iOS 18.2 and it broke"
        matches = []
        for pattern in OS_PATTERNS:
            matches.extend(re.findall(pattern, text, re.IGNORECASE))
        assert any("iOS 18.2" in m for m in matches)

    def test_app_version(self):
        text = "Since updating to v3.2 it crashes"
        matches = []
        for pattern in VERSION_PATTERNS:
            matches.extend(re.findall(pattern, text, re.IGNORECASE))
        assert any("3.2" in m for m in matches)

    def test_app_version_with_patch(self):
        text = "version 3.2.1 is broken"
        matches = []
        for pattern in VERSION_PATTERNS:
            matches.extend(re.findall(pattern, text, re.IGNORECASE))
        assert any("3.2.1" in m for m in matches)

    def test_no_entities(self):
        text = "This app is terrible"
        all_matches = []
        for pattern in DEVICE_PATTERNS + OS_PATTERNS + VERSION_PATTERNS:
            all_matches.extend(re.findall(pattern, text, re.IGNORECASE))
        assert len(all_matches) == 0

    def test_multiple_entities(self):
        text = "Crashes on iPhone 15 and Samsung Galaxy S24 with Android 15 and iOS 18"
        devices = []
        for pattern in DEVICE_PATTERNS:
            devices.extend(re.findall(pattern, text, re.IGNORECASE))
        os_versions = []
        for pattern in OS_PATTERNS:
            os_versions.extend(re.findall(pattern, text, re.IGNORECASE))
        assert len(devices) >= 2
        assert len(os_versions) >= 2


class TestRegexExtract:
    """Test the _regex_extract method."""

    def test_regex_only_mode(self):
        extractor = EntityExtractor(llm_client=None, use_llm=False)
        result = extractor._regex_extract("Crashes on my iPhone 15 Pro with iOS 18.2 after v3.2 update")
        assert "iPhone 15 Pro" in result.devices
        assert "iOS 18.2" in result.os_versions
        assert any("3.2" in v for v in result.app_versions)

    def test_empty_text(self):
        extractor = EntityExtractor(llm_client=None, use_llm=False)
        result = extractor._regex_extract("")
        assert result.devices == []
        assert result.os_versions == []
        assert result.app_versions == []


class TestEntityMerge:
    """Test entity merging and deduplication."""

    def test_merge_deduplicates(self):
        e1 = ExtractedEntities(devices=["iPhone 15"], os_versions=["iOS 18"])
        e2 = ExtractedEntities(devices=["iPhone 15", "Pixel 8"], os_versions=["Android 15"])
        merged = e1.merge(e2)
        assert len(merged.devices) == 2  # iPhone 15 + Pixel 8 (deduped)
        assert "iPhone 15" in merged.devices
        assert "Pixel 8" in merged.devices
        assert len(merged.os_versions) == 2

    def test_merge_empty(self):
        e1 = ExtractedEntities()
        e2 = ExtractedEntities(devices=["Pixel 8"])
        merged = e1.merge(e2)
        assert merged.devices == ["Pixel 8"]

    def test_merge_preserves_all_fields(self):
        e1 = ExtractedEntities(screens=["login"], features=["camera"])
        e2 = ExtractedEntities(screens=["checkout"], features=["dark mode"])
        merged = e1.merge(e2)
        assert "login" in merged.screens
        assert "checkout" in merged.screens
        assert "camera" in merged.features
        assert "dark mode" in merged.features
