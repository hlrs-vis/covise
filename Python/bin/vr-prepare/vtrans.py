#
# Translation functions for vr-prepare
# Visenso GmbH
# (c) 2012
#
# $Id: vtrans.py 785 2014-09-02 08:23:06Z wlukutin $

import gettext
import covise
import os
import sys

covisePath = os.getenv("COVISE_PATH")

vrPrepareDomain = covise.getCoConfigEntry("COVER.Localization.VrPrepareDomain")
localePrefix = covise.getCoConfigEntry("COVER.Localization.LocalePrefix")
languageLocale = covise.getCoConfigEntry("COVER.Localization.LanguageLocale")

if vrPrepareDomain == None:
    vrPrepareDomain = "vr-prepare"

if localePrefix == None:
    localePrefix = "share" + os.sep + "locale"
    
if languageLocale == None:
    languageLocale = "de"
    
print(covisePath)
if covisePath != None and len(covisePath) > 0:
    if covisePath[len(covisePath) - 1] == ';':
        covisePath = covisePath[:-1]
    localePrefix = covisePath + os.sep + localePrefix

print("---------------- LOCALIZATION BEGIN ---------------------------")
print(covisePath)
print(vrPrepareDomain)
print(localePrefix)
print(languageLocale)


from PyQt5 import QtCore, QtGui

class StdOut(object):
    def flush(self):
        sys.__stdout__.flush()
        
    def write(self, string):
#        if isinstance(string, unicode):
#            string = string.encode("utf-8")
        sys.__stdout__.write(string)

sys.stdout = StdOut()
        
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

try:
    trans = gettext.translation( vrPrepareDomain, localePrefix, [ languageLocale ] ) 
    _coTranslate = trans.gettext
except Exception as e:
    print(e)
    print("Unable to initialize localization,")
    print("switching to fall back function.")
    print(vrPrepareDomain)
    print(localePrefix)
    print(languageLocale)
    
    _coTranslate = lambda s: s

'''Bring the functions above together.'''
def coTranslate(s):
    try:
        return _fromUtf8(_coTranslate(s))
    except Exception as e:
        print(e)
        return s
        

print("---------------- LOCALIZATION END -----------------------------")

