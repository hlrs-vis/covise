"""Convert Covise-Config files to new toml dir structure.
"""

__version__ = "1.0"
__author__ = "Marko Djuric"

import os
import argparse
import tomlkit
import xmltodict as xd

DEPRECATED_PLUGINS = "AKToolbar".split()
OVERRIDE = False
ADD = False
ADD_DISABLED = False


def _safe_open(path: str, mode: str, encoding: str="utf-8"):
    """Create dirs before opening file.

    Args:
        path (str): filepath
        mode (str): opening mode
        encoding (str): file encoding

    Returns:
        _type_: return of open
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(file=path, mode=mode, encoding=encoding)


def _get_tml_repr(string: str) -> float | int | str:
    """Helperfunction to check if given string is an integer, decimal or bool and return corresponding python object.

    Args:
        string (str): str to check.

    Returns:
        _type_: Python object as correct type.
    """
    if string.replace(".", "", 1).isdigit():
        return float(string)
    elif string.isdigit() or string.lstrip("-").isdigit():
        return int(string)
    # elif string.lower() in "true false on off".split():
        # return any(string.lower() == valid for valid in "true on".split())
    elif string.lower() in "true false".split():
        return string.lower() == "true"
    return string


def parse_coconfig_list(cc_list: list, parent: str, name: str) -> dict:
    """Convert given coconfig list (xml) to correct toml structure.

    Args:
        cc_list (list): coConfig-XML-List
        parent (str): current parent of the list
        name (str): current key for the list

    Returns:
        dict: toml conform dictionary representation.
    """
    tml_dict = {}
    if parent == "":
        tml_dict[name] = list_repr = {}
    else:
        tml_dict[parent] = {}
        tml_dict[parent][name] = list_repr = {}
    for entry in cc_list:
        if isinstance(entry, dict):
            if all((key in entry.keys() for key in "@name @value".split())):
                list_repr[entry["@name"]] = entry["@value"]
            elif "@name" in entry.keys():
                list_key = entry.pop("@name")
                list_repr[list_key] = create_toml_dict(entry)
            elif "@index" in entry.keys():
                list_key = entry.pop("@index")
                list_repr[list_key] = create_toml_dict(entry)
            else:
                std_str = "Cannot convert " + name
                if parent == "":
                    print(std_str)
                else:
                    print(std_str +" in " + parent + " properly.")
    return tml_dict


def parse_coconfig_dict(cc_dict: dict, name: str) -> dict:
    """Convert given coconfig dictionary (xml) to correct toml structure.

    Args:
        cc_dict (list): coConfig-XML-Dictionary
        name (str): current key for the dict

    Returns:
        dict: toml conform dictionary representation.
    """
    num_elem = len(cc_dict)
    tml_dict = {}
    if num_elem == 0:
        return tml_dict
    elif num_elem == 1:
        if all("@" in k for k, _ in cc_dict.items()):
            tml_dict.update(create_toml_dict(cc_dict, name))
        else:
            tml_dict[name] = create_toml_dict(cc_dict, name)
    else:
        tml_dict[name] = create_toml_dict(cc_dict)

    return tml_dict


def create_toml_dict(coconfig_dict: dict, parent: str = "", skip: list = []) -> dict:
    """Create TOML file as dict.

    Args:
        coconfig_dict (dict): dictionary which contains coconfig entries.
        parent (str, optional): Current parent dictionary of given coconfig_dict (for nested dictionaries). Defaults to "".
        skip (list, optional): Skips entries given in this list. Defaults to [].

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
            tml_dict.update(parse_coconfig_dict(value, key))
        elif isinstance(value, list):
            tml_dict.update(parse_coconfig_list(value, parent, key))

    return tml_dict


def create_toml(coconfig_dict: dict, tml_path: str, manual_adjustment: bool = True, ovrrd: bool = OVERRIDE, add: bool = ADD, skip: list = []) -> None:
    """Create new TOML config.

    Args:
        coconfig_dict (dict): Dictionary which contains coconfig files.
        plug_path (str): Plugin path.
        manual_adjustment (bool): Adjust dictionary according toml structure convention.
        ovrrd (bool): override file
        skip (list): skipping given entries in this list.
    """
    if len(coconfig_dict) == 0:
        return

    dump_dict = coconfig_dict
    if manual_adjustment:
        dump_dict = create_toml_dict(
            coconfig_dict, skip=skip)
        
    if not os.path.exists(tml_path) or ovrrd:
        with _safe_open(tml_path, "wt") as tml:
            tomlkit.dump(dump_dict, tml)
    else:
        with _safe_open(tml_path, mode="rt+", encoding="utf-8") as tml:
            enabled = tomlkit.load(tml).unwrap()
            if enabled == dump_dict:
                return

            new_tml = dump_dict
            if ADD:
                new_tml = dump_dict | enabled
                union_keys = set(enabled) & set(dump_dict)
                for key in union_keys:
                    val = dump_dict[key]
                    if isinstance(val, list):
                        new_tml[key] = list(set(enabled[key] + dump_dict[key]))
                        continue
                    new_tml[key] = val # override always with new value

            if len(new_tml) == 0:
                return
            tml.seek(0)
            tml.truncate(0)
            tomlkit.dump(new_tml, tml)


def create_plugin_toml(coconfig_dict: dict, plug_path: str) -> None:
    """Create new TOML plugin config.

    Args:
        coconfig_dict (dict): Dictionary which contains coconfig files.
        plug_path (str): Plugin path.
    """
    create_toml(coconfig_dict, plug_path, ovrrd=False, skip="@menu @value @shared".split())


def create_module_toml(coconfig_dict: dict, mod_path: str) -> None:
    """Create new TOML module config.

    Args:
        coconfig_dict (dict): Dictionary which contains coconfig files.
        plug_path (str): Plugin path.
    """
    create_toml(coconfig_dict, mod_path)


def create_toplevel_toml(toplevel_config_path: str, values: dict) -> None:
    """Create toplevel TOML in given dir if not already existing.

    Args:
        toplevel_config_path (str): Path to config.
        values (dict): dict holding values for toplevel toml.
    """
    create_toml(values, toplevel_config_path, manual_adjustment=False)


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
    if isinstance(plugins_dict, list):
        for entry in plugins_dict:
            plugin_root_entries.update(iterate_plugins(entry, plugin_rootpath))
    else:
        for plugin_name, plugin_dict in plugins_dict.items():
            if plugin_name in DEPRECATED_PLUGINS or len(plugin_dict) == 0:
                continue
            if isinstance(plugin_dict, list):
                plugin_root_entries.update(
                    iterate_plugins(plugin_dict, plugin_rootpath))
            elif isinstance(plugin_dict, dict):
                for att_name, att_val in plugin_dict.items():
                    for name, li in plugin_root_entries.items():
                        if plugin_name in li: #HACK: won't delete plugins in plugin.toml for now
                            continue
                        at_key = "@" + name
                        if (at_key in att_name and att_val.lower() == "on") or isinstance(att_val, dict):
                            li.append(plugin_name)
                        
                if ADD_DISABLED or plugin_name in plugin_root_entries["value"]:
                    create_plugin_toml(
                        plugin_dict, plugin_rootpath + "/" + plugin_name + ".toml")

    # toplevel value is now load in new config structure
    new_load = plugin_root_entries.pop("value")
    plugin_root_entries["load"] = new_load if not "load" in plugin_root_entries.keys(
    ) else list(set(new_load) | set(plugin_root_entries["load"]))
    return plugin_root_entries


def get_dict(iterable_dict, key: str) -> dict:
    root_global = {}
    if isinstance(iterable_dict, dict):
        root_global = iterable_dict.get(key)
    elif isinstance(iterable_dict, list):
        for d in iterable_dict:
            for k, v in d.items():
                if k in root_global.keys():
                    root_global[k] |= v
                else:
                    root_global[k] = v
    return root_global


def parse_global_section(coconfig: dict, rel_output_path: str) -> None:
    """Convert global section in coconfig to new toml structure.

    Args:
        coconfig (dict): Dictionary which holds entries of current coconfig.
        rel_output_path (str): output path.
    """

    global_key = "GLOBAL"
    if global_key in coconfig.keys():
        opencover_path = rel_output_path + "/opencover"
        covise_path = rel_output_path + "/covise"
        colormap_path = rel_output_path + "/colormaps"
        system_rootpath = rel_output_path + "/system"
        module_rootpath = covise_path + "/modules"
        plugin_rootpath = opencover_path + "/plugins"

        # handle multiple globals
        root_global = get_dict(coconfig, global_key)

        # cover
        cover_dict = get_dict(root_global, "COVER")
        if cover_dict:
            # plugins
            plugins_dict = cover_dict.pop("Plugin", None)
            if plugins_dict != None:
                if not os.path.exists(plugin_rootpath):
                    os.makedirs(plugin_rootpath)

                plugin_root_entries = iterate_plugins(
                    plugins_dict, plugin_rootpath)
                plugin_config_path = opencover_path + "/plugins.toml"
                create_toplevel_toml(plugin_config_path, plugin_root_entries)
            
            # Filemanager
            filemanager_dict = cover_dict.pop("FileManager", None)
            if filemanager_dict != None:
                filemanager_config_path = opencover_path + "/filemanager.toml"
                create_toml(filemanager_dict, filemanager_config_path)

            # general
            create_toml(cover_dict, opencover_path + "/cover.toml")

        # modules
        modules_dict = get_dict(root_global, "Module")
        if modules_dict:
            if not os.path.exists(module_rootpath):
                os.makedirs(module_rootpath)
            module_root_entries = iterate_modules(
                modules_dict, module_rootpath)

            covise_config_path = covise_path + "/modules.toml"
            create_toplevel_toml(covise_config_path, {
                                 "load": module_root_entries})
        
        # colormap
        colormap_dict = get_dict(root_global, "Colormaps")
        if colormap_dict:
            if not os.path.exists(colormap_path):
                os.makedirs(colormap_path)
            
            for name, colordict in colormap_dict.items():
                color_config_path = colormap_path + "/" + name + ".toml"
                create_toml(colordict, color_config_path)

        # system
        system_dict = get_dict(root_global,"System")
        if system_dict:
            if not os.path.exists(system_rootpath):
                os.makedirs(system_rootpath)
            create_toml(system_dict, system_rootpath + "/system.toml")


def parse_coconfig_to_toml(path: str, output_path: str, include: bool = False) -> None:
    """Convert coconfing under path to new TOML structure.

    Args:
        path (str): Path to coconfig xml.
        output_path (str): Output directory.
    """
    tmp_ADD = ADD
    def enable_ADD():
        global ADD
        ADD = True

    def restore_ADD():
        global ADD
        ADD = tmp_ADD

    if include:
        enable_ADD()

    rel_output_path = os.path.realpath(output_dir)
    with open(path, "rb") as f:
        xml_dict = xd.parse(f)

    coconfig = get_dict(xml_dict,"COCONFIG")
    if not len(coconfig):
        print("Please use a config with toplevel domain COCONFIG as tag")
        return

    parse_global_section(coconfig, rel_output_path)

    include_key = "INCLUDE"
    if include_key in coconfig.keys():
        root_include = coconfig[include_key]
        includes = (val.split(".")[0] for include in root_include for key,
                    val in include.items() if "#" in key)
        for include in includes:
            path_to_include = path.rsplit("/", 1)[0] + "/" + include + ".xml"
            parse_coconfig_to_toml(path_to_include, output_path, include=True)

    restore_ADD()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Convert CoConfig into covise conform TOML files.
                                     Note: This script will only load in the coconfig enabled plugins.""")
    parser.add_argument("filepath", help="File path to CoConfig xml")
    parser.add_argument("output", help="Output directory path", default=".")
    parser.add_argument("--override", help="Override plugins.toml without merging. (Good if you disabled plugins)",
                        default=False, action="store_true")
    parser.add_argument("--add", help="Merge only additions to existing plugins.toml.",
                        default=False, action="store_true")
    parser.add_argument("--add_disabled", help="Create TOML files for each COVER plugin mentioned even if they are disabled.",
                        default=False, action="store_true")
    args = parser.parse_args()

    filepath = args.filepath
    output_dir = args.output
    OVERRIDE = args.override
    ADD = args.add
    ADD_DISABLED = args.add_disabled
    parse_coconfig_to_toml(filepath, output_dir)