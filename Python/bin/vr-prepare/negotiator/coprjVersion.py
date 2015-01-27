
"""
initial version number of coprj file
"""

version = 3


"""
History:
    0: old coprj file without versioning (already 0 in default params of coProjectMgr)
    1: projects with versioning (no other important changes)
    2: the full scenegraph is now sent to vr-prepare
       VRML Nodes are now placed below the VRML_VIS
    3: transparency is only sent to leaf nodes (changes on the transparency-panel will be propagated to children)
       NOTE: originalCoprjVersion was introduced with this version (it will be 0 for all older projects)

"""
