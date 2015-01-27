In order to compile the vrmlexp 3ds max plugin with Visual Studio 2005, three environment variables should be set:

3DSMAXINSTALLDIR: The install path to 3ds max. Used to find the stdplugs path where the plugin is copied to. eg. "C:\Program Files\3ds Max 9"
3DSMAXSDK: the path to the maxsdk subdir of the 3ds max SDK install dir. e.g. "C:\Program Files\Autodesk\3ds Max 9 SDK\maxsdk"
3DSMAXLIB: the sub-path for the 3ds max SDK libraries. For 32 Bit it is just "lib". With 64 Bit it should be set to "x64\lib".   
