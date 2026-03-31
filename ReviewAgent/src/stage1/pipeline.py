"""Stage 1 Pipeline: Intake — classification, aspect sentiment, entity extraction."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Callable

from src.common.schemas import ReviewObject
from .classifier import ReviewClassifier
from .aspect_sentiment import AspectSentimentAnalyzer
from .entity_extractor import EntityExtractor
from .hitl_checkpoint import Stage1HITLCheckpoint


class Stage1Pipeline:
    """Orchestrates the full Stage 1 intake process."""

    def __init__(
        self,
        classifier: ReviewClassifier,
        aspect_analyzer: AspectSentimentAnalyzer,
        entity_extractor: EntityExtractor,
        hitl: Stage1HITLCheckpoint | None = None,
    ):
        self.classifier = classifier
        self.aspect_analyzer = aspect_analyzer
        self.entity_extractor = entity_extractor
        self.hitl = hitl or Stage1HITLCheckpoint()

    async def process(self, raw_reviews: list[dict]) -> list[ReviewObject]:
        """Process raw reviews into structured ReviewObjects.

        Args:
            raw_reviews: list of dicts with keys: text, rating, app_id, timestamp
        """
        texts = [r["text"] for r in raw_reviews]

        # Step 1: Classification
        predictions = self.classifier.predict(texts)

        # Step 2: Aspect-based sentiment (async)
        aspects_list = await self.aspect_analyzer.analyze_batch(texts)

        # Step 3: Entity extraction (async)
        entities_list = await self.entity_extractor.extract_batch(texts)

        # Assemble ReviewObjects
        results = []
        for i, raw in enumerate(raw_reviews):
            labels, confidences = predictions[i]
            flagged = self.classifier.needs_hitl(confidences)

            review = ReviewObject(
                review_id=raw.get("review_id", str(uuid.uuid4())),
                text=raw["text"],
                rating=raw.get("rating", 3),
                app_id=raw.get("app_id", "unknown"),
                timestamp=raw.get("timestamp", datetime.now()),
                labels=labels,
                label_confidences=confidences,
                aspects=aspects_list[i],
                entities=entities_list[i],
                flagged_for_hitl=flagged,
            )
            results.append(review)

        return results

    async def process_with_hitl(
        self,
        raw_reviews: list[dict],
        hitl_callback: Callable[[ReviewObject], list[str]] | None = None,
    ) -> list[ReviewObject]:
        """Process reviews with human-in-the-loop verification.

        Args:
            hitl_callback: function that takes a flagged ReviewObject and returns corrected labels.
                          If None, flagged reviews keep their original labels.
        """
        reviews = await self.process(raw_reviews)

        if hitl_callback is None:
            return reviews

        for review in reviews:
            if review.flagged_for_hitl:
                corrected_labels = hitl_callback(review)
                if corrected_labels:
                    self.hitl.record_correction(
                        review_id=review.review_id,
                        original_labels=review.labels,
                        corrected_labels=corrected_labels,
                        rater_id="human",
                    )
                    self.hitl.apply_correction(review, corrected_labels)

        return reviews
