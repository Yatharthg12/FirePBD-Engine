import pytest


def test_zone_area_and_occupancy(simple_building_model):
    zone = simple_building_model.zones["Z0001"]
    assert zone.area == pytest.approx(80.0)
    assert zone.max_occupants >= 1
    assert zone.total_fuel_energy_mj > 0


def test_opening_clear_width(simple_building_model):
    opening = simple_building_model.openings["O001"]
    assert opening.clear_width == pytest.approx(1.4)


def test_building_model_serialization(simple_building_model):
    data = simple_building_model.to_dict()
    assert data["building_id"] == "TEST_BUILDING"
    assert len(data["zones"]) == 3
    assert len(data["openings"]) == 2
    assert data["exit_zone_ids"] == ["Z0003"]
