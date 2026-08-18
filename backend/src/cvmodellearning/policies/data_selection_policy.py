from dataclasses import asdict, dataclass


APPLICATION_DOMAIN_TAGS: dict[str, frozenset[str]] = {
    "traffic": frozenset({"traffic", "street", "autonomous_driving", "driving"}),
    "autonomous driving": frozenset({"traffic", "street", "autonomous_driving", "driving"}),
    "street": frozenset({"traffic", "street", "autonomous_driving", "driving"}),
}


@dataclass(frozen=True)
class DetectionDataSelectionPolicy:
    """Deterministic constraints for detection dataset selection and splitting."""

    policy_id: str = "data.detection.domain_mix.v1"
    preferred_primary_domain_share: float = 0.80
    minimum_primary_domain_share: float = 0.70
    maximum_generalization_sources: int = 1

    def as_context(self) -> dict[str, object]:
        return {
            **asdict(self),
            "domain_share_severity": "warning",
            "domain_share_instruction": (
                "Prefer primary real target-domain data, but a lower share may be used "
                "when the rationale explains domain, availability, or transfer evidence."
            ),
        }


DETECTION_DATA_SELECTION_POLICY = DetectionDataSelectionPolicy()


def application_domain_tags(application_domain: str | None) -> frozenset[str]:
    normalized = (application_domain or "").strip().lower().replace("_", " ")
    if not normalized:
        return frozenset()
    for domain, tags in APPLICATION_DOMAIN_TAGS.items():
        if domain in normalized:
            return tags
    return frozenset()


def matched_domain_tags(
    dataset_domains: tuple[str, ...] | list[str],
    application_domain: str | None,
) -> frozenset[str]:
    requested = application_domain_tags(application_domain)
    return frozenset(str(value).lower() for value in dataset_domains) & requested
