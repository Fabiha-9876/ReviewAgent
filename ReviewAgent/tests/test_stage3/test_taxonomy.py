"""Tests for IssueTaxonomy — template selection and content."""

import pytest
from src.stage3.taxonomy import IssueTaxonomy


class TestIssueTaxonomy:
    """Tests for taxonomy template retrieval."""

    def setup_method(self):
        self.taxonomy = IssueTaxonomy()

    def test_get_template_bug_report(self):
        """Bug report returns Zimmermann template."""
        template = self.taxonomy.get_template("bug_report")
        assert "Zimmermann" in template
        assert "Steps to Reproduce" in template
        assert "Expected Behavior" in template
        assert "Actual Behavior" in template
        assert "Severity" in template

    def test_get_template_feature_request(self):
        """Feature request returns user story template."""
        template = self.taxonomy.get_template("feature_request")
        assert "User Story" in template
        assert "Acceptance Criteria" in template

    def test_get_template_performance(self):
        """Performance returns ISO/IEC 25010 template."""
        template = self.taxonomy.get_template("performance")
        assert "ISO/IEC 25010" in template
        assert "NFR Category" in template
        assert "latency" in template

    def test_get_template_usability(self):
        """Usability returns Nielsen heuristics template."""
        template = self.taxonomy.get_template("usability")
        assert "Nielsen" in template
        assert "10 Usability Heuristics" in template

    def test_get_template_compatibility(self):
        """Compatibility returns device-OS matrix template."""
        template = self.taxonomy.get_template("compatibility")
        assert "Device-OS-Version Matrix" in template
        assert "market share" in template

    def test_get_template_unknown_falls_back_to_bug(self):
        """Unknown issue type falls back to Zimmermann bug template."""
        template = self.taxonomy.get_template("unknown_type")
        bug_template = self.taxonomy.get_template("bug_report")
        assert template == bug_template

    def test_all_templates_are_nonempty_strings(self):
        """All templates return non-empty strings."""
        for issue_type in ["bug_report", "feature_request", "performance", "usability", "compatibility"]:
            template = self.taxonomy.get_template(issue_type)
            assert isinstance(template, str)
            assert len(template) > 50

    def test_all_templates_contain_severity(self):
        """All templates mention severity."""
        for issue_type in ["bug_report", "feature_request", "performance", "usability", "compatibility"]:
            template = self.taxonomy.get_template(issue_type)
            assert "Severity" in template or "severity" in template

    def test_all_templates_contain_affected_component(self):
        """All templates mention affected component."""
        for issue_type in ["bug_report", "feature_request", "performance", "usability", "compatibility"]:
            template = self.taxonomy.get_template(issue_type)
            assert "component" in template.lower()
