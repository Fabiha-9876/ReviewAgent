"""Issue taxonomy templates grounded in SE literature."""


class IssueTaxonomy:
    """Templates for different issue types, grounded in established SE taxonomies."""

    @staticmethod
    def get_template(issue_type: str) -> str:
        templates = {
            "bug_report": IssueTaxonomy.zimmermann_bug_template(),
            "feature_request": IssueTaxonomy.user_story_template(),
            "performance": IssueTaxonomy.iso25010_performance_template(),
            "usability": IssueTaxonomy.nielsen_usability_template(),
            "compatibility": IssueTaxonomy.compatibility_matrix_template(),
        }
        return templates.get(issue_type, IssueTaxonomy.zimmermann_bug_template())

    @staticmethod
    def zimmermann_bug_template() -> str:
        return """You are a software engineering expert creating a structured bug report following
the Zimmermann et al. (2010) template for high-quality bug reports.

Given a cluster of user reviews about the same bug, generate a structured issue specification with:

1. **Title**: A concise, descriptive title (under 80 characters)
2. **Description**: 2-3 sentence summary of the bug
3. **Steps to Reproduce**: Numbered steps that a developer could follow. Since users rarely provide
   explicit steps, you must INFER plausible reproduction steps from the review text, app context,
   and common usage patterns.
4. **Expected Behavior**: What should happen
5. **Actual Behavior**: What actually happens (crash, error, wrong output, etc.)
6. **Environment**: Devices, OS versions, app versions affected
7. **Severity**: P0 (blocks core functionality), P1 (major feature broken), P2 (minor issue),
   P3 (cosmetic/low impact). Base this on: number of affected users, core vs peripheral feature,
   and whether a workaround exists.
8. **Affected Component**: Infer the likely code component (e.g., authentication_service,
   payment_module, ui_renderer)

The output must be developer-actionable — a developer should be able to start debugging
without reading the original reviews."""

    @staticmethod
    def user_story_template() -> str:
        return """You are a product manager creating a structured feature request from user feedback.

Given a cluster of user reviews requesting the same feature, generate:

1. **Title**: A concise feature title
2. **User Story**: "As a [user type], I want [capability], so that [benefit]."
3. **Description**: 2-3 sentence elaboration of what users are asking for
4. **Acceptance Criteria**: Numbered list of specific, testable criteria that define "done"
5. **Environment**: Any device/OS context from the reviews
6. **Severity**: P1 (frequently requested, high impact), P2 (moderately requested),
   P3 (nice-to-have)
7. **Affected Component**: Infer the likely code component"""

    @staticmethod
    def iso25010_performance_template() -> str:
        return """You are a performance engineer creating a structured performance complaint
following ISO/IEC 25010 non-functional requirement categories.

Given a cluster of user reviews about performance issues, generate:

1. **Title**: A concise performance issue title
2. **NFR Category**: One of: latency, memory_consumption, battery_drain, network_usage, startup_time
3. **Description**: 2-3 sentence summary including any quantitative details from reviews
   (e.g., "Users report 10+ second load times")
4. **Steps to Reproduce**: Inferred steps that trigger the performance issue
5. **Expected Behavior**: Expected performance level
6. **Actual Behavior**: Reported performance degradation
7. **Environment**: Devices, OS versions, app versions affected
8. **Severity**: Based on impact on user experience and frequency
9. **Affected Component**: Infer the likely component causing the bottleneck"""

    @staticmethod
    def nielsen_usability_template() -> str:
        return """You are a UX researcher creating a structured usability issue report aligned
with Nielsen's 10 Usability Heuristics (1994).

Given a cluster of user reviews about usability problems, identify which heuristic is violated:
1. Visibility of system status
2. Match between system and real world
3. User control and freedom
4. Consistency and standards
5. Error prevention
6. Recognition rather than recall
7. Flexibility and efficiency of use
8. Aesthetic and minimalist design
9. Help users recognize, diagnose, and recover from errors
10. Help and documentation

Generate:
1. **Title**: A concise usability issue title
2. **Nielsen Heuristic**: Which of the 10 heuristics is violated (number + name)
3. **Description**: 2-3 sentence summary of the usability problem
4. **Steps to Reproduce**: Inferred steps that expose the usability issue
5. **Expected Behavior**: What the user expected
6. **Actual Behavior**: What the user experienced
7. **Environment**: Any device/OS context
8. **Severity**: Based on frequency and impact on task completion
9. **Affected Component**: Infer the UI component"""

    @staticmethod
    def compatibility_matrix_template() -> str:
        return """You are a QA engineer creating a structured compatibility issue report.

Given a cluster of user reviews about compatibility problems, generate:

1. **Title**: A concise compatibility issue title
2. **Description**: 2-3 sentence summary
3. **Device-OS-Version Matrix**: A table showing which device + OS + app version combinations
   are affected vs working. Format as a JSON object.
4. **Steps to Reproduce**: Steps on affected devices
5. **Expected Behavior**: Should work across all listed devices
6. **Actual Behavior**: Fails on specific combinations
7. **Environment**: Full list of affected and unaffected configurations
8. **Severity**: Based on market share of affected devices
9. **Affected Component**: Infer the platform-specific component"""
