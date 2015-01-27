#! /usr/bin/env python

import os
import sys
import shutil

BASE_DIR = os.getcwd() #sys.path[0]
REFERENCE_DIR = BASE_DIR + "/reference"
TMP_DIR = BASE_DIR + "/tmp"




def getDirectoryList(dir):
    tmplist = os.listdir(dir)
    return [s for s in tmplist if not os.path.isfile(dir + "/" + s)]
    
def getAllDirectories(subdir):
    tmp = getDirectoryList(TMP_DIR + "/" + subdir)
    result = []
    for item in tmp:
        newsubs = getDirectoryList(TMP_DIR + "/" + subdir + item)
        if (newsubs == []):
            result.append(subdir + item)
        else:
            for item in tmp:
                result.extend(getAllDirectories(subdir + item + "/"))
    return result

def isValidItem(elementSelection, item):
    if ("v" in elementSelection) and item.endswith(".covise"):
        return True
    if ("e" in elementSelection) and ("numberOfExecutes" in item):
        return True
    if ("s" in elementSelection) and item.endswith("_000.png"): # dont copy DIFF files
        return True
    return False








dirlist = getAllDirectories("")
dirlist.sort()
selection = []






print
print("Available projects:")
for i in range(len(dirlist)):
    print(" ", str(i).zfill(2), dirlist[i])
print
print("="*80)







while True:
    print
    print("Please insert the indices or a part of the directories name (for example '3 stroemung 28 29')")
    print("Insert '*' to select all projects.")
    print("You can do this several times. Selecting again removes a directory. When you're done, just press Return.")
    raw_indices = raw_input("-> ")
    if (raw_indices == ""):
        break
    for tok in raw_indices.split(" "):
        try:
            # check index first
            val = int(tok)
            dir = dirlist[val]
            if dir in selection:
                selection.remove(dir)
            else:
                selection.append(dir)
            break
        except:
            # maybe part of the string
            for dir in dirlist:
                if (tok == "*") or (tok.lower() in dir.lower()):
                    if dir in selection:
                        selection.remove(dir)
                    else:
                        selection.append(dir)
    print
    print("Current selection:")
    if (selection == []):
        print("  EMPTY")
    for dir in selection:
        print(" ", dir)

if (selection == []):
    sys.exit(0)







print
print("="*80)
print
print("Your selection:")
for dir in selection:
    print(" ", dir)






print
print("="*80)
print
print("Please select the elements you want to replace (for example: 'vs')")
print("v = VisItems")
print("e = Number of Executes")
print("s = Screenshots")
elementSelection = raw_input("-> ")

if (elementSelection == ""):
    sys.exit(0)






print
print("Continue (y/n)?")
if (raw_input("-> ") != "y"):
    sys.exit(0)




# GO
print
print("="*80)
print
for dir in selection:
    print("Copying", dir, "...")
    allTargetFiles = os.listdir(REFERENCE_DIR + "/" + dir)
    for item in allTargetFiles:
        if isValidItem(elementSelection, item):
            os.remove(REFERENCE_DIR + "/" + dir + "/" + item)
    allSourceFiles = os.listdir(TMP_DIR + "/" + dir)
    for item in allSourceFiles:
        if isValidItem(elementSelection, item):
            shutil.copy(TMP_DIR + "/" + dir + "/" + item, REFERENCE_DIR + "/" + dir + "/" + item)
print

