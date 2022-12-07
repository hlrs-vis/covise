Configure COVISE for VS Code
----------------------------

This program injects options in ./settings.json and ./launch.json in ../settings.json and ../launch.json
This requires #nlohmann/json from <https://github.com/nlohmann/json.git>

The program is meant to be build and executed through the VS Code configure COVISE task when you are working with COVISE for the first time.

Troubleshooting:
CMake can not find nlohmann/json: make sure it is installed globally or in externlibs/vcpkg and the dependency path is set correctly

The program fails to parse your json files: VS Code json interpreter is more forgiving than nlohmann/json. Make sure you do not have trailing commas in sour settings.json and launch.json  

The Terminal environment is not set: close all integrated terminals and restart VS Code or run the "Terminal :Kill All Terminals " and "Developer: Reload Window" commands

CMake can not find cl.exe (Windows only): Check covise.env in this folder for a PAth and PATH entry. If there are both merge them to PATH
