"""Pydantic models for the Samvahan medicine redistribution system."""

from enum import Enum
from pydantic import BaseModel


class FacilityType(str, Enum):
    district_store = "district_store"
    hospital = "hospital"
    chc = "chc"
    phc = "phc"


class DrugTier(str, Enum):
    life_saving = "life_saving"
    essential = "essential"
    routine = "routine"


TIER_CRITICALITY = {
    DrugTier.life_saving: 3,
    DrugTier.essential: 2,
    DrugTier.routine: 1,
}


class StorageCondition(str, Enum):
    room_temp = "room_temp"
    cold_2_8c = "cold_2_8c"


class Facility(BaseModel):
    id: str
    name: str
    type: FacilityType
    latitude: float
    longitude: float
    has_cold_storage: bool


class Drug(BaseModel):
    code: str
    name: str
    molecule: str
    strength: str
    form: str
    tier: DrugTier
    criticality: int
    requires_cold_chain: bool
    unit_price_inr: float


class Lot(BaseModel):
    lot_id: str
    facility_id: str
    drug_code: str
    batch_number: str
    quantity: int
    days_to_expiry: int
    storage_condition: StorageCondition


class Route(BaseModel):
    source_id: str
    destination_id: str
    transit_days: int
    trip_cost_inr: float
    has_cold_transport: bool


class UsageRate(BaseModel):
    facility_id: str
    drug_code: str
    units_per_day: float


class Reserve(BaseModel):
    facility_id: str
    drug_code: str
    min_stock: int


class TransferVariable(BaseModel):
    var_id: str
    chunk_index: int
    lot_id: str
    source_id: str
    destination_id: str
    drug_code: str
    chunk_quantity: int
    usable_quantity: int
    days_to_expiry: int
    transit_days: int
    value_at_risk: float
    transport_cost: float


class TransferOrder(BaseModel):
    order_id: str
    lot_id: str
    source_id: str
    destination_id: str
    drug_code: str
    quantity: int
    batch_number: str
    days_to_expiry: int
    justification: str


class ValidationResult(BaseModel):
    is_valid: bool
    violations: list[str]
    warnings: list[str]
    fefo_deviations: list[str] = []


class SurplusChunk(BaseModel):
    chunk_id: str
    chunk_index: int
    lot_id: str
    source_facility_id: str
    drug_code: str
    quantity: int
    days_to_expiry: int
    storage_condition: str
    batch_number: str
    value_at_risk: float


class InventorySnapshot(BaseModel):
    needs: dict[str, int]
    surplus_chunks: list[SurplusChunk]
    facility_stock: dict[str, int]


class InventoryData(BaseModel):
    facilities: dict[str, Facility]
    drugs: dict[str, Drug]
    lots: list[Lot]
    routes: list[Route]
    usage_rates: dict[str, UsageRate]
    reserves: dict[str, Reserve]
    route_map: dict[str, Route]
    stock_by_facility_drug: dict[str, int]
    needs: dict[str, int]
