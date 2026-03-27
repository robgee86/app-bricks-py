import ctypes
import fcntl
import os
import re


MEDIA_ENT_ID_FLAG_NEXT = 1 << 31
MEDIA_ENT_F_CAM_SENSOR = 0x20001
MEDIA_LNK_FL_IMMUTABLE = 1 << 1


class MediaEntityInfo(ctypes.Union):
    _fields_ = [("raw", ctypes.c_uint8 * 184)]


class MediaEntityDesc(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("name", ctypes.c_char * 32),
        ("type", ctypes.c_uint32),
        ("revision", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("group_id", ctypes.c_uint32),
        ("pads", ctypes.c_uint16),
        ("links", ctypes.c_uint16),
        ("reserved", ctypes.c_uint32 * 4),
        ("info", MediaEntityInfo),
    ]


class MediaPadDesc(ctypes.Structure):
    _fields_ = [
        ("entity", ctypes.c_uint32),
        ("index", ctypes.c_uint16),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 2),
    ]


class MediaLinkDesc(ctypes.Structure):
    _fields_ = [
        ("source", MediaPadDesc),
        ("sink", MediaPadDesc),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 2),
    ]


class MediaLinksEnum(ctypes.Structure):
    _fields_ = [
        ("entity", ctypes.c_uint32),
        ("pads", ctypes.POINTER(MediaPadDesc)),
        ("links", ctypes.POINTER(MediaLinkDesc)),
        ("reserved", ctypes.c_uint32 * 4),
    ]


def _iowr(type_char, nr, size):
    return (3 << 30) | (size << 16) | (ord(type_char) << 8) | nr


MEDIA_IOC_ENUM_ENTITIES = _iowr("|", 0x01, ctypes.sizeof(MediaEntityDesc))
MEDIA_IOC_ENUM_LINKS = _iowr("|", 0x02, ctypes.sizeof(MediaLinksEnum))


def find_sensor_i2c_addr(media_dev, csiphy_index):
    """
    Traverse the media graph to find CSIPHY a sensor with an immutable link to the provided CSIPHY.
    Return the I2C address of the found sensor.
    """
    fd = os.open(media_dev, os.O_RDWR)
    try:
        # Enumerate all entities
        entities = []
        desc = MediaEntityDesc()
        desc.id = MEDIA_ENT_ID_FLAG_NEXT
        while True:
            try:
                fcntl.ioctl(fd, MEDIA_IOC_ENUM_ENTITIES, desc)
            except OSError:
                break
            entities.append({
                "id": desc.id,
                "name": desc.name.decode(errors="ignore").rstrip("\x00"),
                "type": desc.type,
                "num_pads": desc.pads,
                "num_links": desc.links,
            })
            desc.id |= MEDIA_ENT_ID_FLAG_NEXT

        by_id = {e["id"]: e for e in entities}
        csiphy_name = f"msm_csiphy{csiphy_index}"

        # For each sensor, look for the IMMUTABLE link to its CSIPHY
        for entity in entities:
            if entity["type"] != MEDIA_ENT_F_CAM_SENSOR:
                continue
            if entity["num_links"] == 0:
                continue

            pads = (MediaPadDesc * entity["num_pads"])()
            links = (MediaLinkDesc * entity["num_links"])()
            req = MediaLinksEnum()
            req.entity = entity["id"]
            req.pads = pads
            req.links = links
            fcntl.ioctl(fd, MEDIA_IOC_ENUM_LINKS, req)

            for i in range(entity["num_links"]):
                if not (links[i].flags & MEDIA_LNK_FL_IMMUTABLE):
                    continue
                sink = by_id.get(links[i].sink.entity)
                if sink and sink["name"] == csiphy_name:
                    m = re.search(r"(\d+-[\da-fA-F]{4})", entity["name"])
                    if m:
                        return m.group(1)

        raise RuntimeError(f"No sensor found on {csiphy_name}")
    finally:
        os.close(fd)