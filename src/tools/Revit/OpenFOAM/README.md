OpenFOAMInterface for Autodesk Revit
=====================================================

OpenFOAMInterface is an Addin for Autodesk Revit used for simulate the indoor airflow in OpenFOAM based on the designed HVAC system in the scene. 
The plugin uses the BIM data stored in Revit to calculate the boundaries necessary for the simulation.

License
-------

OpenFOAMInterface source code is licensed under the LGPL v2.1. See `lgpl-2.1.txt` for
details.

Build Requirements
------------------

- **.NET**
  at least 4.5
  
- **CMake**:
  at least 3.0

- **Revit**:
  Autodesk Revit needs to be installed for the dependencies RevitAPI.dll and RevitAPIUI.dll (default Folder <Program Data>/Autodesk/Revit <version>)

Building OpenFOAMInterface with cmake-gui and visual studio 2019
---------------

Create a subdirectory for building, change to it, and invoke CMake-GUI:

      cmake-gui ..
	  
Configure and generate the CMakeFiles. Invoke the Visual Studio environment via `Open Project` in cmake-gui or with commandline command `devenv`.
Add RevitAPI.dll and RevitAPIUI.dll to current session with right click on OpenFOAMInterface > Depenency in the Project-Explorer and select `Add dependencies`.
Navigate to `Browse` and add both libs from the install directory of your Revit Version. After that choose to build as `Release` and rightclick on OpenFOAMInterface in the Project-Explorer and select `Build`.

Install Addin
---------------

Copy OpenFOAMInterface.addin from the directory `<sourcedir>/BIM/Resources` and OpenFOAMInterface.dll from your default build output folder (depending on you Visual Studio Setup it's `C:\opt\lib\`) to `C:\Users\<Username>\AppData\Roaming\Autodesk\Revit\Addins\<Version>`.

Your good to go. Revit will prompt you at startup to choose if you wanna load the addin always.

Have fun.

Usage
---------------

- TODO

Source Code Organization
------------------------

- `Resources`
  all resources

- `BIM`:
  source code

    - `BIM/OpenFOAM`: OpenFOAM parameters, constants and dicts
    - `BIM/OpenFOAMUI`: GUI classes for settings (deprecated)
	- `BIM/Properties`: ResourceManager
	- `BIM/Resources`: Common used resources like addin
