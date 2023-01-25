"""Convert Covise-Config files to new toml dir structure.
"""

__version__ = "1.0"
__author__ = "Marko Djuric"

import os
import argparse
import tomli_w
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import xmltodict as xd

DEPRECATED_PLUGINS = "AKToolbar".split()
OVERRIDE = False
ADD = False


def _get_key(config_dict, key): return config_dict.get(key)


def _get_tml_repr(string: str):
    """Helperfunction to check if given string is an integer, decimal or bool and return corresponding python object.

    Args:
        string (str): str to check.

    Returns:
        _type_: Python object as correct type.
    """
    if string.isdigit():
        return int(string)
    elif string.isdecimal():
        return float(string)
    elif string.lower() in ("true", "false"):
        return bool(string)
    elif string.lower() in ("on", "off"):
        return string.lower() == "on"
    return string


def create_toml_dict(coconfig_dict: dict, parent: str = "", skip: list = []) -> dict:
    """Create TOML file as string representation.

    Args:
        coconfig_dict (dict): dictionary which contains coconfig entries.
        parent (str, optional): Current parent dictionary of given coconfig_dict (for nested dictionaries). Defaults to "".
        skip (list, optional): Skips entries given in this list.

    Returns:
        dict: TOML file as dict.
    """
    tml_dict = {}
    for key, value in coconfig_dict.items():
        if "@" in key:
            if key in skip and parent == "":
                continue
            att_name = key.split("@")[1] if parent == "" else parent
            tml_dict[att_name] = _get_tml_repr(value)
        elif isinstance(value, dict):
            if len(value) <= 1:
                tml_dict.update(create_toml_dict(value, key))
            else:
                tml_dict[key] = create_toml_dict(value)
        elif isinstance(value, list):
            if parent == "":
                list_repr = tml_dict[key] = {}
            else:
                tml_dict[parent] = {}
                list_repr = tml_dict[parent][key] = {}

            for entry in value:
                if isinstance(entry, dict):
                    # HACK: for now workaround for modules but needs more generic approach to work with complex configs like config-midi.xml
                    if all((key in entry.keys() for key in "@name @value".split())):
                        list_repr[entry["@name"]] = entry["@value"]

    return tml_dict


def create_toml(coconfig_dict: dict, tml_path: str, skip: list = []) -> None:
    """Create new TOML config.

    Args:
        coconfig_dict (dict): Dictionary which contains coconfig files.
        plug_path (str): Plugin path.
        skip (list): skipping given entries in this list.
    """
    with open(tml_path, "wb") as tml:
        tomli_w.dump(create_toml_dict(
            coconfig_dict, skip=skip), tml)


def create_plugin_toml(coconfig_dict: dict, plug_path: str) -> None:
    """Create new TOML plugin config.

    Args:
        coconfig_dict (dict): Dictionary which contains coconfig files.
        plug_path (str): Plugin path.
    """
    create_toml(coconfig_dict, plug_path, skip="@menu @value @shared".split())


def create_module_toml(coconfig_dict: dict, mod_path: str) -> None:
    """Create new TOML module config.

    Args:
        coconfig_dict (dict): Dictionary which contains coconfig files.
        plug_path (str): Plugin path.
    """
    create_toml(coconfig_dict, mod_path)


def create_toplevel_toml(toplevel_config_path: str, values: dict) -> None:
    """Create toplevel TOML in plugin dir if not already existing.

    Args:
        toplevel_config_path (str): Path to config.
        values (dict): dict holding values for toplevel toml.
    """
    if not os.path.exists(toplevel_config_path) or OVERRIDE:
        with open(toplevel_config_path, mode="wb") as tml:
            tomli_w.dump(values, tml)
    else:
        with open(toplevel_config_path, mode="wb+") as tml:
            enabled = tomllib.load(tml)
            new_tml = values | enabled if ADD else enabled | values  # union of dicts
            tomli_w.dump(new_tml, tml)


def iterate_modules(modules_dict: dict, module_rootpath: str) -> list:
    """Iterate through modules and create toml for each.

    Args:
        modules_dict (dict): Contains all modules as dictionary.
        module_rootpath (str): Output directory for toml files.

    Returns:
        list: Enabled modules.
    """
    modules_to_load = []
    for module_name, module_dict in modules_dict.items():
        modules_to_load.append(module_name)
        create_module_toml(module_dict, module_rootpath +
                           "/" + module_name + ".toml")

    return modules_to_load


def iterate_plugins(plugins_dict: dict, plugin_rootpath: str) -> dict:
    """Iterate through plugins dict and create toml for enabled plugins.

    Args:
        plugins_dict (dict): dictionary which contains plugins
        plugin_rootpath (str): opencover plugin dir
    """
    load, menu, shared = ([] for i in range(3))

    # dicitionary holding entries for root plugin.toml
    plugin_root_entries = dict(
        zip("value menu shared".split(), [load, menu, shared]))
    for plugin_name, plugin_dict in plugins_dict.items():
        if plugin_name in DEPRECATED_PLUGINS:
            continue
        if isinstance(plugin_dict, list):
            # TODO: add list handling => for now ignore
            continue
        for att_name, att_val in plugin_dict.items():
            for name, li in plugin_root_entries.items():
                at_key = "@" + name
                if at_key in att_name and _get_tml_repr(att_val):
                    li.append(plugin_name)
                elif isinstance(att_val, dict):
                    create_plugin_toml(
                        plugin_dict, plugin_rootpath + "/" + plugin_name + ".toml")

    # toplevel value is now load in new config structure
    plugin_root_entries["load"] = plugin_root_entries.pop("value")
    return plugin_root_entries


def parse_global_section(coconfig: dict, rel_output_path: str) -> None:
    """Convert global section in coconfig to new toml structure.

    Args:
        coconfig (dict): Dictionary which holds entries of current coconfig.
        rel_output_path (str): output path.
    """

    global_key = "GLOBAL"
    if global_key in coconfig.keys():
        root_global = coconfig[global_key]
        opencover_path = rel_output_path + "/opencover"
        covise_path = rel_output_path + "/covise"
        system_rootpath = rel_output_path + "/system"
        module_rootpath = covise_path + "/modules"
        plugin_rootpath = opencover_path + "/plugins"

        # plugins
        cover_dict = _get_key(root_global, "COVER")
        if cover_dict:
            plugins_dict = _get_key(cover_dict, "Plugin")
            cover_dict.pop("Plugin")
            create_toml(cover_dict, opencover_path + "/cover.toml")

            if not os.path.exists(plugin_rootpath):
                os.makedirs(plugin_rootpath)

            plugin_root_entries = iterate_plugins(
                plugins_dict, plugin_rootpath)
            plugin_config_path = opencover_path + "/plugins.toml"
            create_toplevel_toml(plugin_config_path, plugin_root_entries)

        # modules
        modules_dict = _get_key(root_global, "Module")
        if modules_dict:
            if not os.path.exists(module_rootpath):
                os.makedirs(module_rootpath)
            module_root_entries = iterate_modules(
                modules_dict, module_rootpath)

            covise_config_path = covise_path + "/modules.toml"
            create_toplevel_toml(covise_config_path, {
                                 "load": module_root_entries})

        # system
        system_dict = _get_key(root_global, "System")
        if system_dict:
            if not os.path.exists(system_rootpath):
                os.makedirs(system_rootpath)
            create_toml(system_dict, system_rootpath + "/system.toml")


def parse_coconfig_to_toml(path: str, output_path: str) -> None:
    """Convert coconfing under path to new TOML structure.

    Args:
        path (str): Path to coconfig xml.
        output_path (str): Output directory.
    """
    rel_output_path = os.path.realpath(output_dir)
    keypath_start = "COCONFIG"
    with open(path, "rb") as f:
        xml_dict = xd.parse(f)

    coconfig = xml_dict["COCONFIG"]
    parse_global_section(coconfig, rel_output_path)

    include_key = "INCLUDE"
    if include_key in coconfig.keys():
        root_include = coconfig[include_key]
        includes = (val.split(".")[0] for include in root_include for key,
                    val in include.items() if "#" in key)
        # TODO: handle includes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Convert CoConfig into covise conform TOML files.
                                     Note: This script will only load in the coconfig enabled plugins.""")
    parser.add_argument("filepath", help="File path to CoConfig xml")
    parser.add_argument("output", help="Output directory path", default=".")
    parser.add_argument("--override", help="Override plugins.toml without merging. (Good if you disabled plugins)",
                        default=False, action="store_true")
    parser.add_argument("--add", help="Merge only additions to existing plugins.toml.",
                        default=False, action="store_true")
    args = parser.parse_args()

    filepath = args.filepath
    output_dir = args.output
    OVERRIDE = args.override
    ADD = args.add
    parse_coconfig_to_toml(filepath, output_dir)
