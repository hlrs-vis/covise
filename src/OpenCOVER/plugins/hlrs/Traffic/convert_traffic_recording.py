#!/usr/bin/env python3
"""
WARNING: This script was entirely vibe-coded. Don't trust it.

Convert a SUMO FCD XML file to a compact little-endian binary format. Used with
`ConnectorTrafficRecording` to render traffic without simulating it at the same
time. Allows seeking through the animation slider.

Usage:
    sumo -c path/to/sumo.sumocfg --step-length 0.2 --begin 3600 --end 7200 --fcd-output simulation.fcd.xml
    python convert_traffic_recording.py simulation.fcd.xml simulation.traffic.bin

The converter performs two XML passes:

1. Collect all vehicle IDs.
2. Write the binary output and build a timestep offset index.

Vehicle indexes are assigned by lexicographically sorting all unique vehicle
IDs. This makes the indexes deterministic for a given set of IDs and
independent of the order in which vehicles first appear in the XML.

Binary format
=============

All integers and floating-point values are little-endian.

Header:
    4 bytes    magic bytes: b"FCD1"
    uint32     format version
    uint32     number of vehicles
    uint32     number of timesteps
    uint64     absolute byte offset of the timestep index

Vehicle dictionary:
    Repeated once for each vehicle:
        uint16   UTF-8 vehicle ID length
        bytes    UTF-8 vehicle ID

Timestep data:
    Repeated for each timestep:
        float64  simulation time
        uint32   number of vehicle records

Vehicle record:
    Repeated for each vehicle in a timestep:
        uint32   stable vehicle index
        float32  x coordinate
        float32  y coordinate
        float32  z coordinate
        float32  angle
        float32  speed

Each vehicle record is 24 bytes. The original vehicle ID is stored only once
in the vehicle dictionary. A record's vehicle index refers to the corresponding
entry in that dictionary.

Timestep index:
    Repeated once for each timestep, in timestep order:
        uint64   absolute byte offset of the timestep data

The first index entry points to the first timestep. To seek directly to
timestep N, read the uint64 value at:

    index_offset + N * 8

Then seek to that byte offset and read the timestep header.

Missing x, y, z, angle, or speed attributes are encoded as IEEE-754 NaN.

The script produces no terminal output.
"""

from __future__ import annotations

import argparse
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

MAGIC = b"FCD1"
VERSION = 1

# magic, version, vehicle_count, timestep_count, index_offset
HEADER = struct.Struct("<4sIIIQ")

STRING_LENGTH = struct.Struct("<H")
TIMESTEP_HEADER = struct.Struct("<dI")
VEHICLE_RECORD = struct.Struct("<Ifffff")
INDEX_ENTRY = struct.Struct("<Q")


def collect_vehicle_ids(xml_path: Path) -> list[str]:
    """Collect and deterministically sort all vehicle IDs."""
    vehicle_ids: set[str] = set()

    for _, element in ET.iterparse(xml_path, events=("end",)):
        if element.tag == "vehicle":
            vehicle_id = element.get("id")

            if vehicle_id is not None:
                vehicle_ids.add(vehicle_id)

            element.clear()

    return sorted(vehicle_ids)


def parse_float(
    element: ET.Element,
    attribute: str,
    default: float = math.nan,
) -> float:
    """Parse a floating-point XML attribute."""
    value = element.get(attribute)

    if value is None:
        return default

    return float(value)


def write_string(output, value: str) -> None:
    """Write a uint16-length-prefixed UTF-8 string."""
    encoded = value.encode("utf-8")

    if len(encoded) > 65535:
        raise ValueError(
            f"Vehicle ID exceeds the maximum length of 65535 bytes: {value!r}"
        )

    output.write(STRING_LENGTH.pack(len(encoded)))
    output.write(encoded)


def convert(xml_path: Path, binary_path: Path) -> None:
    vehicle_ids = collect_vehicle_ids(xml_path)

    vehicle_to_index = {
        vehicle_id: index for index, vehicle_id in enumerate(vehicle_ids)
    }

    timestep_count = 0
    timestep_offsets: list[int] = []

    with binary_path.open("w+b") as output:
        # Write a placeholder header. The timestep count and index offset
        # are patched after all timesteps have been written.
        output.write(
            HEADER.pack(
                MAGIC,
                VERSION,
                len(vehicle_ids),
                0,
                0,
            )
        )

        for vehicle_id in vehicle_ids:
            write_string(output, vehicle_id)

        for _, element in ET.iterparse(xml_path, events=("end",)):
            if element.tag != "timestep":
                continue

            time_text = element.get("time")

            if time_text is None:
                raise ValueError("A timestep is missing its 'time' attribute.")

            simulation_time = float(time_text)

            vehicles = [child for child in element if child.tag == "vehicle"]

            # Store the absolute file offset before writing this timestep.
            timestep_offsets.append(output.tell())

            output.write(
                TIMESTEP_HEADER.pack(
                    simulation_time,
                    len(vehicles),
                )
            )

            for vehicle in vehicles:
                vehicle_id = vehicle.get("id")

                if vehicle_id is None:
                    raise ValueError("A vehicle element is missing its 'id' attribute.")

                vehicle_index = vehicle_to_index[vehicle_id]

                x = parse_float(vehicle, "x")
                y = parse_float(vehicle, "y")
                z = parse_float(vehicle, "z")
                angle = parse_float(vehicle, "angle")
                speed = parse_float(vehicle, "speed")

                output.write(
                    VEHICLE_RECORD.pack(
                        vehicle_index,
                        x,
                        y,
                        z,
                        angle,
                        speed,
                    )
                )

            timestep_count += 1
            element.clear()

        # The timestep index is appended after all timestep data.
        index_offset = output.tell()

        for timestep_offset in timestep_offsets:
            output.write(INDEX_ENTRY.pack(timestep_offset))

        # Patch the header with the final timestep count and index offset.
        output.seek(0)
        output.write(
            HEADER.pack(
                MAGIC,
                VERSION,
                len(vehicle_ids),
                timestep_count,
                index_offset,
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert SUMO FCD XML to compact binary."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    convert(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
