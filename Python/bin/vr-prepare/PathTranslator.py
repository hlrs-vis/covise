#
# Translation functions for vr-prepare
# Visenso GmbH
# (c) 2012
#
# $Id: PathTranslator.py 704 2012-11-20 10:27:29Z wlukutin $

import os

def convert_to_os_path(path):
    retpath = path
    retpath = retpath.replace("\\", os.sep)
    retpath = retpath.replace("/", os.sep)
    return retpath

def strip_locale(loc):
    ret_loc = convert_to_os_path(loc)
    
    idx = ret_loc.rfind('.')
    idxsep = ret_loc.rfind("." + os.sep)
    if idx > 0 and idx != idxsep:
        ret_loc = ret_loc[:idx]
    
    idx = ret_loc.rfind('@')
    if idx > 0:
        ret_loc = ret_loc[:idx]
    
    return ret_loc
    
def translate_path(loc, path):
    local = strip_locale(loc)
    
    if len(local) <= 0:
        return path
    
    retpath = convert_to_os_path(path)
    
    sepp = retpath.rfind(os.sep)
    
    if sepp >= 0:
        leftpath = retpath[:sepp]
        rightpath = retpath[sepp:]
        retpath = leftpath + os.sep + local + rightpath
    
    if os.path.isfile(retpath):
        return retpath
    else:
        return path
