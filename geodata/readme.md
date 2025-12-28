## 1) Amenities density (shops, healthcare, schools, etc.)

**API:** OpenStreetMap **Overpass API**

* Query POIs around a street segment (buffer) using tags like `amenity=*`, `shop=*`, `healthcare=*`, `tourism=*`. [OpenStreetMap**+1**](https://wiki.openstreetmap.org/wiki/Overpass_API?utm_source=chatgpt.com)

## 2) Transit accessibility (stops, stations)

**API options:**

* **Overpass API** : `public_transport=platform`, `highway=bus_stop`, `railway=station`, etc. [OpenStreetMap**+1**](https://wiki.openstreetmap.org/wiki/Overpass_API?utm_source=chatgpt.com)
* **Transitland** (API token, global GTFS/GTFS-RT aggregation): useful when you prefer GTFS-based stop locations rather than OSM tags. [transit.land**+1**](https://www.transit.land/documentation/datasets/downloading/?utm_source=chatgpt.com)
* **Germany-specific** : `gtfs.de` (GTFS + GTFS-RT offerings) and local networks like **VBB GTFS-RT** for Berlin/Brandenburg. [gtfs.de**+2**gtfs.de**+2**](https://gtfs.de/en/?utm_source=chatgpt.com)

## 3) Green proximity (parks, forests, green areas)

**API:** Overpass API

* Tags: `leisure=park`, `natural=wood`, `landuse=forest/grass`, `leisure=playground`, etc. [OpenStreetMap**+1**](https://wiki.openstreetmap.org/wiki/Overpass_API?utm_source=chatgpt.com)

## 4) Traffic / noise proxy (road type, speed limits, major roads nearby)

**API:** Overpass API

* Use the street’s own `highway=*` class (e.g., `primary/secondary/residential`) and nearby road classes as a proxy; optionally include `maxspeed=*`, `lanes=*` where present. [OpenStreetMap**+1**](https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL?utm_source=chatgpt.com)

  Note: this is a  *proxy* , not real-time traffic.

## 5) Nightlife density (bars, clubs)

**API:** Overpass API

* Tags: `amenity=bar`, `amenity=pub`, `amenity=nightclub`, etc. [OpenStreetMap**+1**](https://wiki.openstreetmap.org/wiki/Overpass_API/Language_Guide?utm_source=chatgpt.com)

## 6) Air quality (PM2.5, NO2, etc.)

**API options:**

* **OpenAQ API v3** (global; requires API key header `X-API-Key`). [OpenAQ Docs**+1**](https://docs.openaq.org/using-the-api/api-key?utm_source=chatgpt.com)
* **Germany** : Umweltbundesamt (UBA) **Air Data API** (REST API; official DE air monitoring). [Luftdaten**+1**](https://luftdaten.umweltbundesamt.de/en?utm_source=chatgpt.com)
* **Europe** : European Environment Agency  **Air Quality Download Service API** . [Europäische Umweltagentur](https://www.eea.europa.eu/en/datahub/datahubitem-view/778ef9f5-6293-4846-badd-56a29c70880d?utm_source=chatgpt.com)
