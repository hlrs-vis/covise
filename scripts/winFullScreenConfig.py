import argparse
import ctypes
import re
import xml.etree.ElementTree as ET
from ctypes import wintypes

# usage: winFullScreenConfig.py [display_selector xml_file]
# If no arguments are given, prints all display geometries. 
# Otherwise, updates the given config XML file with the geometry of the selected display. 
# The display selector can be a numeric display ID or one of: primary, secondary, prefer-secondary. 
# The script uses Windows API calls to query display information and modify the XML config accordingly.

DISPLAY_DEVICE_ATTACHED_TO_DESKTOP = 0x00000001
DISPLAY_DEVICE_PRIMARY_DEVICE = 0x00000004
ENUM_CURRENT_SETTINGS = -1
CCHDEVICENAME = 32
CCHFORMNAME = 32


class POINTL(ctypes.Structure):
    _fields_ = [
        ("x", wintypes.LONG),
        ("y", wintypes.LONG),
    ]


class DISPLAY_DEVICEW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("DeviceName", wintypes.WCHAR * 32),
        ("DeviceString", wintypes.WCHAR * 128),
        ("StateFlags", wintypes.DWORD),
        ("DeviceID", wintypes.WCHAR * 128),
        ("DeviceKey", wintypes.WCHAR * 128),
    ]


class DEVMODEW(ctypes.Structure):
    _fields_ = [
        ("dmDeviceName", wintypes.WCHAR * CCHDEVICENAME),
        ("dmSpecVersion", wintypes.WORD),
        ("dmDriverVersion", wintypes.WORD),
        ("dmSize", wintypes.WORD),
        ("dmDriverExtra", wintypes.WORD),
        ("dmFields", wintypes.DWORD),
        ("dmPosition", POINTL),
        ("dmDisplayOrientation", wintypes.DWORD),
        ("dmDisplayFixedOutput", wintypes.DWORD),
        ("dmColor", wintypes.SHORT),
        ("dmDuplex", wintypes.SHORT),
        ("dmYResolution", wintypes.SHORT),
        ("dmTTOption", wintypes.SHORT),
        ("dmCollate", wintypes.SHORT),
        ("dmFormName", wintypes.WCHAR * CCHFORMNAME),
        ("dmLogPixels", wintypes.WORD),
        ("dmBitsPerPel", wintypes.DWORD),
        ("dmPelsWidth", wintypes.DWORD),
        ("dmPelsHeight", wintypes.DWORD),
        ("dmDisplayFlags", wintypes.DWORD),
        ("dmDisplayFrequency", wintypes.DWORD),
        ("dmICMMethod", wintypes.DWORD),
        ("dmICMIntent", wintypes.DWORD),
        ("dmMediaType", wintypes.DWORD),
        ("dmDitherType", wintypes.DWORD),
        ("dmReserved1", wintypes.DWORD),
        ("dmReserved2", wintypes.DWORD),
        ("dmPanningWidth", wintypes.DWORD),
        ("dmPanningHeight", wintypes.DWORD),
    ]


def get_display_id(device_name):
    match = re.search(r"DISPLAY(\d+)$", device_name)
    if match:
        return int(match.group(1))
    return None


def get_displays():
    displays = []
    index = 0

    while True:
        device = DISPLAY_DEVICEW()
        device.cb = ctypes.sizeof(DISPLAY_DEVICEW)
        if not ctypes.windll.user32.EnumDisplayDevicesW(None, index, ctypes.byref(device), 0):
            break

        if not (device.StateFlags & DISPLAY_DEVICE_ATTACHED_TO_DESKTOP):
            index += 1
            continue

        devmode = DEVMODEW()
        devmode.dmSize = ctypes.sizeof(DEVMODEW)
        if ctypes.windll.user32.EnumDisplaySettingsW(device.DeviceName, ENUM_CURRENT_SETTINGS, ctypes.byref(devmode)):
            displays.append({
                "display_id": get_display_id(device.DeviceName),
                "name": device.DeviceName,
                "label": device.DeviceString,
                "left": devmode.dmPosition.x,
                "top": devmode.dmPosition.y,
                "width": devmode.dmPelsWidth,
                "height": devmode.dmPelsHeight,
                "right": devmode.dmPosition.x + devmode.dmPelsWidth,
                "bottom": devmode.dmPosition.y + devmode.dmPelsHeight,
                "primary": bool(device.StateFlags & DISPLAY_DEVICE_PRIMARY_DEVICE),
            })

        index += 1

    return displays


def find_display(displays, display_id):
    for display in displays:
        if display["display_id"] == display_id:
            return display
    return None


def resolve_display(displays, selector):
    normalized = selector.strip().lower()

    if normalized == "primary":
        for display in displays:
            if display["primary"]:
                return display
        return None

    if normalized == "secondary":
        secondary_displays = [display for display in displays if not display["primary"]]
        if len(secondary_displays) == 1:
            return secondary_displays[0]
        if not secondary_displays:
            return None
        raise SystemExit("More than one secondary monitor found. Use the numeric display ID instead.")

    if normalized == "prefer-secondary":
        secondary_displays = [display for display in displays if not display["primary"]]
        if len(secondary_displays) == 1:
            return secondary_displays[0]
        if len(secondary_displays) > 1:
            raise SystemExit("More than one secondary monitor found. Use the numeric display ID instead.")

        for display in displays:
            if display["primary"]:
                return display
        return None

    try:
        display_id = int(selector)
    except ValueError as error:
        raise SystemExit(
            "Display selector must be a numeric display ID or one of: primary, secondary, prefer-secondary."
        ) from error

    return find_display(displays, display_id)


def update_window_config(xml_path, display):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    window = root.find(".//WindowConfig/Window")
    if window is None:
        raise ValueError("No <Window> element found inside <WindowConfig>.")

    window.set("width", str(display["width"]))
    window.set("height", str(display["height"]))
    window.set("left", str(display["left"]))
    window.set("top", str(display["top"]))

    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def print_displays(displays, primary):
    selected = [display for display in displays if display["primary"] is primary]
    if not selected:
        print("No primary monitor found." if primary else "No secondary monitor found.")
        return

    for index, display in enumerate(selected, 1):
        if primary:
            label = f"Primary monitor {index}" if len(selected) > 1 else "Primary monitor"
        else:
            label = f"Secondary monitor {index}" if len(selected) > 1 else "Secondary monitor"

        print(f"{label}:")
        if display["display_id"] is not None:
            print(f"  Display  : {display['display_id']}")
        print(f"  Device   : {display['name']} ({display['label']})")
        print(f"  Position : ({display['left']}, {display['top']})")
        print(f"  Size     : {display['width']} x {display['height']} pixels")
        print(
            f"  Rect     : left={display['left']}, top={display['top']}, "
            f"right={display['right']}, bottom={display['bottom']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print Windows display geometry or apply a display geometry to a COVISE XML config."
    )
    parser.add_argument(
        "display_selector",
        nargs="?",
        help="Windows display number or one of: primary, secondary, prefer-secondary",
    )
    parser.add_argument("xml_file", nargs="?", help="Path to the XML config file to modify")
    return parser.parse_args()


def main():
    args = parse_args()
    all_displays = get_displays()

    if args.display_selector is None and args.xml_file is None:
        print_displays(all_displays, primary=False)
        print_displays(all_displays, primary=True)
        return

    if args.display_selector is None or args.xml_file is None:
        raise SystemExit("Provide both display_selector and xml_file, or provide neither.")

    display = resolve_display(all_displays, args.display_selector)
    if display is None:
        raise SystemExit(f"Display {args.display_selector} not found.")

    update_window_config(args.xml_file, display)
    print(f"Updated {args.xml_file} for display {args.display_selector}.")
    print(f"  Position : ({display['left']}, {display['top']})")
    print(f"  Size     : {display['width']} x {display['height']} pixels")


if __name__ == "__main__":
    main()