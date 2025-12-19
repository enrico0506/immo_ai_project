from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from immo_ai.utils.text import first_present, parse_float


class ExposeRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str | None = None
    expose_id: str | None = None
    title: str | None = None
    address: str | None = None
    description: str | None = None
    ausstattung_text: str | None = None
    lage_text: str | None = None
    features: list[str] | None = None
    summary: dict[str, str] | None = None
    costs: dict[str, str] | None = None
    details: dict[str, str] | None = None
    energy: dict[str, str] | None = None
    coordinates: dict[str, float] | None = None
    immobilie1_id: str | None = None
    anbieter_id: str | None = None
    images: list[str] | None = None
    image_hashes: list[str] | None = None

    price_eur: float | None = None
    area_sqm: float | None = None
    rooms: float | None = None

    source: str
    run_id: str | None = None
    fetched_at: str | None = None
    html_sha256: str | None = None

    @model_validator(mode="after")
    def normalize_fields(self) -> "ExposeRecord":
        if self.price_eur is None:
            price_text = _find_value(self, ["kaufpreis", "preis", "gesamtpreis", "kaltmiete"])
            self.price_eur = parse_float(price_text)
        if self.area_sqm is None:
            area_text = _find_value(self, ["wohnflaeche", "wohnflache", "flaeche", "wohnfl\u00e4che"])
            self.area_sqm = parse_float(area_text)
        if self.rooms is None:
            rooms_text = _find_value(self, ["zimmer", "zimmeranzahl", "anzahl_zimmer"])
            self.rooms = parse_float(rooms_text)
        return self


def _find_value(record: ExposeRecord, keys: list[str]) -> str | None:
    sources: list[dict[str, str] | None] = [record.summary, record.costs, record.details]
    for source in sources:
        if not source:
            continue
        for key in keys:
            for source_key, value in source.items():
                if key in source_key:
                    return value
    return first_present([record.summary.get(keys[0]) if record.summary else None])
