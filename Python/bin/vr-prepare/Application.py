
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

class _VRPApplication(object):

    """Global accessable thing for essential data."""

    def __init__(self):
        self.visuKey2GuiKey = {} # Use e.g. when user changes visibility in cover to syncronize with gui.
        self.guiKey2visuKey = {} # An item in gui may have attached a different item for visualization.

        self.key2type = {} # For getting the type of an object by the global key.
        self.key2params = {} # Key onto parameters
        self.mw = None # for reference to the mainwindow
        self.globalJournalMgrParams = None
        self.globalJournalMgrKey = -5

        
_theVrpApp = None
def vrpAppI():
    """Access the singleton."""
    global _theVrpApp
    if None == _theVrpApp: _theVrpApp = _VRPApplication()
    return _theVrpApp
    
vrpApp = vrpAppI()
    
# eof
