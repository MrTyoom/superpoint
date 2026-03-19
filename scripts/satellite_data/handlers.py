import math
import os
import sys
import uuid

import geojson
import osmium
import shapely.geometry


class FeatureStorage:
    """Stores features on disk and handles batching.

       Note: you have to call flush at the end to flush the last partial batch.
    """

    def __init__(self, out, batch):
        assert batch > 0

        self.out = out
        self.batch = batch

        self.features = []

    def add(self, feature):
        if len(self.features) >= self.batch:
            self.flush()

        self.features.append(feature)

    def flush(self):
        if not self.features:
            return

        collection = geojson.FeatureCollection(self.features)

        base, ext = os.path.splitext(self.out)
        suffix = uuid.uuid4().hex

        out = "{}-{}{}".format(base, suffix, ext)

        with open(out, "w") as fp:
            geojson.dump(collection, fp)

        self.features.clear()


def is_polygon(way):
    """Checks if the way is a polygon.

    Args
      way: the osmium.osm.Way to check.

    Returns:
      True if the way is a polygon, False otherwise.

    Note: The geometry shape can still be invalid (e.g. self-intersecting).
    """

    if not way.is_closed():
        return False

    if len(way.nodes) < 4:
        return False

    return True


class BuildingHandler(osmium.SimpleHandler):
    """Extracts building polygon features (visible in satellite imagery) from the map.
    """

    # building=* to discard because these features are not vislible in satellite imagery
    building_filter = set(
        ["construction", "houseboat", "static_caravan", "stadium", "conservatory", "digester", "greenhouse", "ruins"]
    )

    # location=* to discard because these features are not vislible in satellite imagery
    location_filter = set(["underground", "underwater"])

    def __init__(self, out, batch):
        super().__init__()
        self.storage = FeatureStorage(out, batch)

    def way(self, w):
        if not is_polygon(w):
            return

        if "building" not in w.tags:
            return

        if w.tags["building"] in self.building_filter:
            return

        if "location" in w.tags and w.tags["location"] in self.location_filter:
            return

        geometry = geojson.Polygon([[(n.lon, n.lat) for n in w.nodes]])
        shape = shapely.geometry.shape(geometry)

        if shape.is_valid:
            feature = geojson.Feature(geometry=geometry)
            self.storage.add(feature)
        else:
            print("Warning: invalid feature: https://www.openstreetmap.org/way/{}".format(w.id), file=sys.stderr)

    def flush(self):
        self.storage.flush()

class ParkingHandler(osmium.SimpleHandler):
    """Extracts parking lot polygon features (visible in satellite imagery) from the map.
    """

    # parking=* to discard because these features are not vislible in satellite imagery
    parking_filter = set(["underground", "sheds", "carports", "garage_boxes"])

    def __init__(self, out, batch):
        super().__init__()
        self.storage = FeatureStorage(out, batch)

    def way(self, w):
        if not is_polygon(w):
            return

        if "amenity" not in w.tags or w.tags["amenity"] != "parking":
            return

        if "parking" in w.tags:
            if w.tags["parking"] in self.parking_filter:
                return

        geometry = geojson.Polygon([[(n.lon, n.lat) for n in w.nodes]])
        shape = shapely.geometry.shape(geometry)

        if shape.is_valid:
            feature = geojson.Feature(geometry=geometry)
            self.storage.add(feature)
        else:
            print("Warning: invalid feature: https://www.openstreetmap.org/way/{}".format(w.id), file=sys.stderr)

    def flush(self):
        self.storage.flush()


class RoadHandler(osmium.SimpleHandler):
    """Extracts road polygon features (visible in satellite imagery) from the map.
    """

    highway_attributes = {
        "motorway": {
            "lanes": 4,
            "lane_width": 3.75,
            "left_hard_shoulder_width": 0.75,
            "right_hard_shoulder_width": 3.0,
        },
        "trunk": {"lanes": 3, "lane_width": 3.75, "left_hard_shoulder_width": 0.75, "right_hard_shoulder_width": 3.0},
        "primary": {
            "lanes": 2,
            "lane_width": 3.75,
            "left_hard_shoulder_width": 0.50,
            "right_hard_shoulder_width": 1.50,
        },
        "secondary": {
            "lanes": 1,
            "lane_width": 3.50,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.75,
        },
        "tertiary": {
            "lanes": 1,
            "lane_width": 3.50,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.75,
        },
        "unclassified": {
            "lanes": 1,
            "lane_width": 3.50,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.00,
        },
        "residential": {
            "lanes": 1,
            "lane_width": 3.50,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.75,
        },
        "service": {
            "lanes": 1,
            "lane_width": 3.00,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.00,
        },
        "motorway_link": {
            "lanes": 2,
            "lane_width": 3.75,
            "left_hard_shoulder_width": 0.75,
            "right_hard_shoulder_width": 3.00,
        },
        "trunk_link": {
            "lanes": 2,
            "lane_width": 3.75,
            "left_hard_shoulder_width": 0.50,
            "right_hard_shoulder_width": 1.50,
        },
        "primary_link": {
            "lanes": 1,
            "lane_width": 3.50,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.75,
        },
        "secondary_link": {
            "lanes": 1,
            "lane_width": 3.50,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.75,
        },
        "tertiary_link": {
            "lanes": 1,
            "lane_width": 3.50,
            "left_hard_shoulder_width": 0.00,
            "right_hard_shoulder_width": 0.00,
        },
    }

    road_filter = set(highway_attributes.keys())

    EARTH_MEAN_RADIUS = 6371004.0

    def __init__(self, out, batch):
        super().__init__()
        self.storage = FeatureStorage(out, batch)

    def way(self, w):
        if "highway" not in w.tags:
            return

        if w.tags["highway"] not in self.road_filter:
            return

        left_hard_shoulder_width = self.highway_attributes[w.tags["highway"]]["left_hard_shoulder_width"]
        lane_width = self.highway_attributes[w.tags["highway"]]["lane_width"]
        lanes = self.highway_attributes[w.tags["highway"]]["lanes"]
        right_hard_shoulder_width = self.highway_attributes[w.tags["highway"]]["right_hard_shoulder_width"]

        if "oneway" not in w.tags:
            lanes = lanes * 2
        elif w.tags["oneway"] == "no":
            lanes = lanes * 2

        if "lanes" in w.tags:
            try:
                # Roads have at least one lane; guard against data issues.
                lanes = max(int(w.tags["lanes"]), 1)

                # Todo: take into account related lane tags
                # https://wiki.openstreetmap.org/wiki/Tag:busway%3Dlane
                # https://wiki.openstreetmap.org/wiki/Tag:cycleway%3Dlane
                # https://wiki.openstreetmap.org/wiki/Key:parking:lane
            except ValueError:
                print("Warning: invalid feature: https://www.openstreetmap.org/way/{}".format(w.id), file=sys.stderr)

        road_width = left_hard_shoulder_width + lane_width * lanes + right_hard_shoulder_width

        if "width" in w.tags:
            try:
                # At least one meter wide, for road classes specified above
                road_width = max(float(w.tags["width"]), 1.0)

                # Todo: handle optional units such as "2 m"
                # https://wiki.openstreetmap.org/wiki/Key:width
            except ValueError:
                print("Warning: invalid feature: https://www.openstreetmap.org/way/{}".format(w.id), file=sys.stderr)

        geometry = geojson.LineString([(n.lon, n.lat) for n in w.nodes])
        shape = shapely.geometry.shape(geometry)
        geometry_buffer = shape.buffer(math.degrees(road_width / 2.0 / self.EARTH_MEAN_RADIUS))

        if shape.is_valid:
            feature = geojson.Feature(geometry=shapely.geometry.mapping(geometry_buffer))
            self.storage.add(feature)
        else:
            print("Warning: invalid feature: https://www.openstreetmap.org/way/{}".format(w.id), file=sys.stderr)

    def flush(self):
        self.storage.flush()

class WaterBodyHandler(osmium.SimpleHandler):
    """Extracts water body polygon features (visible in satellite imagery) from the map."""
    water_filter = set(["reservoir", "pond", "lake", "riverbank", "basin", "wetland", "marsh", "stream"])

    def __init__(self, out, batch):
        super().__init__()
        self.storage = FeatureStorage(out, batch)
        self._wkb_factory = osmium.geom.WKBFactory()

    def way(self, w):
        if not is_polygon(w):
            return
        if ("natural" in w.tags and w.tags["natural"] == "water") or ("water" in w.tags):
            water_type = w.tags.get("water") or w.tags.get("natural")
            if water_type and water_type not in self.water_filter and water_type not in {"water", "lake", "reservoir", "pond", "riverbank"}:
                pass
        else:
            return

        try:
            geometry = geojson.Polygon([[(n.lon, n.lat) for n in w.nodes]])
        except Exception:
            print(f"Warning: failed to build polygon for way {w.id}", file=sys.stderr)
            return

        shape = shapely.geometry.shape(geometry)

        if shape.is_valid and not shape.is_empty:
            properties = dict(w.tags)
            feature = geojson.Feature(geometry=geometry, properties=properties)
            self.storage.add(feature)
        else:
            print(f"Warning: invalid feature: https://www.openstreetmap.org/way/{w.id}", file=sys.stderr)

    def relation(self, r):
        if r.tags.get("type") != "multipolygon":
            return

        if not (r.tags.get("natural") == "water" or "water" in r.tags):
            return

        try:
            wkb = self._wkb_factory.create_multipolygon(r)
        except Exception:
            return

        if not wkb:
            return

        try:
            shape = shapely.wkb.loads(wkb, hex=False)
        except Exception:
            print(f"Warning: invalid WKB for relation {r.id}", file=sys.stderr)
            return

        if shape.is_empty or not shape.is_valid:
            print(f"Warning: invalid feature: https://www.openstreetmap.org/relation/{r.id}", file=sys.stderr)
            return

        geom_mapping = shapely.geometry.mapping(shape)
        properties = dict(r.tags)
        feature = geojson.Feature(geometry=geom_mapping, properties=properties)
        self.storage.add(feature)

    def flush(self):
        self.storage.flush()


class FieldHandler(osmium.SimpleHandler):
    """Extracts agricultural field polygons (arable, meadow, pasture, crop, etc.) from OSM."""

    field_keys = set(["landuse", "natural", "agricultural", "landcover"])
    field_values = {
        "landuse": set(["farmland", "grass", "meadow", "orchard", "vineyard", "farmyard"]),
        "natural": set(["grassland"]),
        "agricultural": set(["crop", "arable"]),
        "landcover": set(["grass", "crops"]),
    }

    location_filter = set(["underground", "underwater"])

    def __init__(self, out, batch):
        super().__init__()
        self.storage = FeatureStorage(out, batch)
        self._wkb_factory = osmium.geom.WKBFactory()

    def _is_field(self, tags):
        if not tags:
            return False
        td = dict(tags)
        if "location" in td and td["location"] in self.location_filter:
            return False
        for k in self.field_keys:
            if k in td:
                v = td[k]
                if k in self.field_values and v in self.field_values[k]:
                    return True
                if k == "landuse" and v in ("grass", "meadow", "farmland"):
                    return True
        return False

    def _publish_shape(self, shape, tags, osm_type, osm_id):
        if shape is None or shape.is_empty:
            print(f"Warning: empty feature: https://www.openstreetmap.org/{osm_type}/{osm_id}", file=sys.stderr)
            return

        if not shape.is_valid:
            try:
                fixed = shape.buffer(0)
                if fixed.is_empty or not fixed.is_valid:
                    print(f"Warning: invalid feature after fix: https://www.openstreetmap.org/{osm_type}/{osm_id}", file=sys.stderr)
                    return
                shape = fixed
            except Exception:
                print(f"Warning: invalid feature: https://www.openstreetmap.org/{osm_type}/{osm_id}", file=sys.stderr)
                return

        geom_mapping = shapely.geometry.mapping(shape)
        properties = dict(tags) if tags else {}
        feature = geojson.Feature(geometry=geom_mapping, properties=properties)
        self.storage.add(feature)

    def way(self, w):
        if not is_polygon(w):
            return
        tags = dict(w.tags)
        if not self._is_field(tags):
            return
        try:
            poly = geojson.Polygon([[(n.lon, n.lat) for n in w.nodes]])
            shape = shapely.geometry.shape(poly)
        except Exception:
            print(f"Warning: failed geometry: https://www.openstreetmap.org/way/{w.id}", file=sys.stderr)
            return
        self._publish_shape(shape, tags, "way", w.id)

    def relation(self, r):
        if r.tags.get("type") != "multipolygon":
            return
        tags = dict(r.tags)
        if not self._is_field(tags):
            return
        try:
            wkb = self._wkb_factory.create_multipolygon(r)
        except Exception:
            return
        if not wkb:
            return

        try:
            shape = shapely.wkb.loads(wkb, hex=False)
        except Exception:
            print(f"Warning: invalid WKB for relation: https://www.openstreetmap.org/relation/{r.id}", file=sys.stderr)
            return

        self._publish_shape(shape, tags, "relation", r.id)

    def flush(self):
        self.storage.flush()
