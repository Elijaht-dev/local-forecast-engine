"""Version 1 input contract. Values are indexed by UTC interval start."""

from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

MAX_HISTORY = 8640  # 90 days at 15-minute resolution
MAX_HORIZON = 672
Slug = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")]
Number = FiniteFloat | None
Vector = Annotated[list[Number], Field(max_length=MAX_HISTORY)]
FutureVector = Annotated[list[Number], Field(max_length=MAX_HORIZON)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Covariate(StrictModel):
    key: Slug
    history: Vector
    future: FutureVector | None = None
    received_at: int | None = Field(default=None, ge=0, le=4102444800)


class Solar(StrictModel):
    kind: Literal["power", "energy"]
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class Dataset(StrictModel):
    project_id: Slug
    series_key: Slug
    config_key: Slug = "default"
    unit: str = Field(max_length=48)
    timezone: str = Field(default="UTC", max_length=64)
    step_minutes: Literal[15, 60]
    cutoff: int = Field(ge=0, le=4102444800)
    horizon: int = Field(ge=1, le=MAX_HORIZON)
    target: Vector = Field(min_length=1)
    covariates: list[Covariate] = Field(default_factory=list, max_length=16)
    solar: Solar | None = None

    @property
    def step(self) -> int:
        return self.step_minutes * 60

    @property
    def start(self) -> int:
        return self.cutoff - len(self.target) * self.step

    @model_validator(mode="after")
    def validate_grid(self):
        if self.start < 0 or self.cutoff % self.step or self.horizon * self.step_minutes > 10080:
            raise ValueError(
                "The cutoff must align with the interval and the horizon is at most 7 days"
            )
        try:
            ZoneInfo(self.timezone)
        except (ValueError, ZoneInfoNotFoundError) as error:
            raise ValueError("Expected an IANA time zone") from error
        if self.solar and self.unit != {"power": "W", "energy": "Wh"}[self.solar.kind]:
            raise ValueError("Solar input must be normalized to W or Wh per interval")
        seen = set()
        for covariate in self.covariates:
            if covariate.key in seen:
                raise ValueError("Covariate keys must be unique")
            if len(covariate.history) != len(self.target):
                raise ValueError("Historical covariates must align with the target")
            if covariate.future is not None and (
                len(covariate.future) != self.horizon or covariate.received_at is None
            ):
                raise ValueError("Future inputs require an aligned horizon and receipt timestamp")
            seen.add(covariate.key)
        return self


class ForecastRequest(StrictModel):
    dataset: Dataset
    model: Slug
    publish: bool = False
    use_inputs: bool = True


class Variant(StrictModel):
    model: Slug
    use_inputs: bool = True


class BacktestRequest(StrictModel):
    dataset: Dataset
    variants: list[Variant] = Field(min_length=1, max_length=10)
    folds: int = Field(default=5, ge=1, le=10)
