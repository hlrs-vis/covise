#! /usr/bin/env python

# maximum number of executes allowed to differ during loading for a warning only
EXECUTES_LOADING_WARNING_MAX = 5
# maximum number of executes allowed to differ during presentation steps for a warning only
EXECUTES_STEPS_WARNING_MAX = 1
# maximum amount of bytes allowed to differ for a warning only
VISITEM_WARNING_MAX = 32
# maximum number of pixels allowed to differ more than 1% for a warning only
SNAPSHOT_WARNING_MAX = 25

# set to True to create reference files
CREATE_EXECUTES_REFERENCE = 0
CREATE_VISITEMS_REFERENCE = 0
CREATE_SNAPSHOT_REFERENCE = 0

# files (without subdirectory) to be tested (should be [".*\.coprj"] but you can change it e.g. for debugging)
TEST_FILES_REGEX = [".*\.coprj"]

#TEST_FILES_REGEX = ["Stabmagnet_klein\.coprj"]
#TEST_FILES_REGEX = ["msport_blue_red_trans_pressure_trace52_512d_cut_vec_iso_path\.coprj"]
#TEST_FILES_REGEX = ["CC_.*\.coprj"]
#TEST_FILES_REGEX = ["CC_0291311020000000_Instrumente_Leistenbruch_OPCC_0110609090000000_Tutorial.coprj"]
#TEST_FILES_REGEX = [".*tiny.*\.coprj"]
#TEST_FILES_REGEX = ["tiny_all_3D_features\.coprj"]
#TEST_FILES_REGEX = ["testNecker.coprj"]


# timeout in minutes
TIMEOUT = 15

# commands
VR_PREPARE = "vr-prepare"
IM_COMPARE = "compare"
CLEAN_UP = ["clean_covise", "clean_cover"]

######################################################################



import sys
import os
import subprocess
import fileinput
import time
import datetime
import signal
import killableprocess
import re



######################################################################
#                       helper functions
######################################################################

def deleteFiles(dir, fileEnding):
    for file in os.listdir(dir):
        if file.endswith(fileEnding):
            os.remove(dir + '/' + file)

def safeMakeDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def safeDelete(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)

def fileWrite(filepath, content):
    file = open(filepath, "w")
    file.write(str(content))
    file.close()

def fileRead(filepath):
    file = open(filepath, "r")
    result = file.readline()
    file.close()
    return result

def getFileListMatchingRegex(dir, regex):
    pattern = re.compile(regex + "$")
    tmplist = os.listdir(dir)
    return [s for s in tmplist if os.path.isfile(dir + "/" + s) and pattern.match(s)]

def getFileListMatchingRegexList(dir, regexList):
    tmplist = os.listdir(dir)
    result = []
    for regex in regexList:
        pattern = re.compile(regex + "$")
        for file in tmplist:
            if os.path.isfile(dir + "/" + file) and pattern.match(file) and (not file in result):
                result.append(file)
    return result

def getDirectoryList(dir):
    tmplist = os.listdir(dir)
    return [s for s in tmplist if not os.path.isfile(dir + "/" + s)]

#def getFileSize(filepath):
    #statinfo = os.stat(filepath)
    #return statinfo.st_size

def getFileDiff(filepath1, filepath2):
    file1 = open(filepath1, "r")
    file2 = open(filepath2, "r")
    diff = 0
    while True:
        c1 = file1.read(1)
        c2 = file2.read(1)
        if (c1 == "") and (c2 == ""):
            break
        if (c1 != c2):
            diff = diff + 1
    file1.close()
    file2.close()
    return diff

def runVrPrepare(project, logfile):
    # run
    log = open(logfile, "w")
    process = killableprocess.Popen([VR_PREPARE, project], stdout=log, stderr=log)
    ret = process.wait(TIMEOUT*60, True)
    log.close()
    return (ret != -9) # -9 means timeout

def cleanUp():
    for cmd in CLEAN_UP:
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()

def getProcesses():
    output = subprocess.Popen(["ps","-ef"], stdout = subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0] # we only need stdout
    for line in output.split("\n"):
        yield line.replace("\r", "")

def imageMagickCompare(image1, image2, outImage):
    try:
        process = subprocess.Popen([IM_COMPARE, "-fuzz", "1%", "-metric", "AE", image1, image2, outImage], stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        output = process.communicate()[0] # we only need stdout
        return int(output.split()[0]) # the first entry is the value we want
    except: #if the images are totally different we get "compare: ImagesTooDissimilar ...", so casting to int throws an exception
        return 999999







# our directories
BASE_DIR = os.getcwd() #sys.path[0]
PROJECT_DIR = BASE_DIR + "/coprj"
CONFIG_DIR = BASE_DIR + "/config"
IGNORE_DIR = BASE_DIR + "/ignore"
REFERENCE_DIR = BASE_DIR + "/reference"
TMP_DIR = BASE_DIR + "/tmp"
GLOBAL_LOG = BASE_DIR + "/logfile.txt"

# output
safeDelete(GLOBAL_LOG)
outfile = open(GLOBAL_LOG, "w")
def output(line):
    outfile.write(line + "\n")
    # immediatelly write to disk so the progress can be monitored
    outfile.flush()
    os.fsync(outfile.fileno())








######################################################################
#                     main test function
######################################################################

def test(currentFilenameWithSubdir):

    # convenience variables
    currentBasenameWithSubdir, extension = currentFilenameWithSubdir.split(".")
    currentProjectFile = PROJECT_DIR + '/' + currentFilenameWithSubdir
    currentIgnoreFile = IGNORE_DIR + '/' + currentBasenameWithSubdir + '.txt'
    currentTmpDir = TMP_DIR + "/" + currentBasenameWithSubdir
    currentRefDir = REFERENCE_DIR + "/" + currentBasenameWithSubdir
    currentLogfile = currentTmpDir + "/logfile.txt"
    if CREATE_EXECUTES_REFERENCE:
        currentExecutesLoadingFile = currentRefDir + "/numberOfExecutesLoading.txt"
        currentExecutesStepsFile = currentRefDir + "/numberOfExecutesSteps.txt"
    else:
        currentExecutesLoadingFile = currentTmpDir + "/numberOfExecutesLoading.txt"
        currentExecutesStepsFile = currentTmpDir + "/numberOfExecutesSteps.txt"

    output("-"*100)
    output("Testing: " + currentBasenameWithSubdir)

    # prepare directories and set environment variables
    safeMakeDir(currentTmpDir)
    if CREATE_VISITEMS_REFERENCE or CREATE_EXECUTES_REFERENCE or CREATE_SNAPSHOT_REFERENCE:
        safeMakeDir(currentRefDir)
    safeDelete(currentLogfile)
    if CREATE_VISITEMS_REFERENCE:
        os.environ["VR_PREPARE_DEBUG_VISITEMS_DIR"] = currentRefDir
        deleteFiles(currentRefDir, ".covise")
    else:
        os.environ["VR_PREPARE_DEBUG_VISITEMS_DIR"] = currentTmpDir
        deleteFiles(currentTmpDir, ".covise")
    if CREATE_EXECUTES_REFERENCE:
        safeDelete(currentRefDir + "/numberOfExecutesLoading.txt")
        safeDelete(currentRefDir + "/numberOfExecutesSteps.txt")
    if CREATE_SNAPSHOT_REFERENCE:
        os.environ["VR_PREPARE_DEBUG_SNAPSHOTS_DIR"] = currentRefDir
        deleteFiles(currentRefDir, ".png")
    else:
        os.environ["VR_PREPARE_DEBUG_SNAPSHOTS_DIR"] = currentTmpDir
        deleteFiles(currentTmpDir, ".png")

    # config
    configName = currentBasenameWithSubdir + '.xml'
    if not os.path.exists(CONFIG_DIR + '/' + configName): configName = "default.xml"
    os.environ["COCONFIG"] = CONFIG_DIR + '/' + configName

    # run
    runOK = runVrPrepare(currentProjectFile, currentLogfile)
    cleanUp()

    # result: check console errors
    if not runOK:
        output("Console   ERROR: Timeout")
        return
    errors = [
              "Traceback",
              "Segmentation fault",
              "Removing the module",
              "license failed",
              "coConfigRootErrorHandler::error"
              ]
    errorFound = False
    for line in fileinput.input(currentLogfile):
        for error in errors:
            if error in line:
                errorFound = True
                break
        if errorFound:
            break
    fileinput.close()
    if errorFound:
        output("Console   ERROR")
        return # dont check anything else if we have errors in the console
    output("Console   OK")
    
    # load a list of files to ignore (if existing)
    ignoreList = []
    if os.path.exists(currentIgnoreFile):
        for line in file(currentIgnoreFile):
            ignoreList.append(line.replace("\n", "").replace("\r", ""))

    # result: check number of executes
    executeCount = [0, 0]
    switch = 0
    for line in fileinput.input(currentLogfile):
        if line.startswith(">>> pre"):
            executeCount[switch] = executeCount[switch] + 1
        if line.startswith("+++++ TEST_VISITEMS +++++ Select presentation step"):
            switch = 1
    fileinput.close()
    fileWrite(currentExecutesLoadingFile, executeCount[0])
    fileWrite(currentExecutesStepsFile, executeCount[1])
    if CREATE_EXECUTES_REFERENCE:
        output("Executes  WARNING: reference created")
    else:
        if not os.path.exists(currentRefDir + "/numberOfExecutesLoading.txt") or not os.path.exists(currentRefDir + "/numberOfExecutesSteps.txt"):
            output("Executes  ERROR: reference files do not exist")
        else:
            referenceCount = [int(fileRead(currentRefDir + "/numberOfExecutesLoading.txt")), int(fileRead(currentRefDir + "/numberOfExecutesSteps.txt"))]
            if (executeCount != referenceCount):
                diff = [abs(x-y) for x,y in zip(executeCount, referenceCount)]
                if (diff[0] > EXECUTES_LOADING_WARNING_MAX) or (diff[1] > EXECUTES_STEPS_WARNING_MAX):
                    errorOrWarning = "ERROR"
                else:
                    errorOrWarning = "WARNING"
                output("Executes  " + errorOrWarning + ": " + str(executeCount[0]) + "+" + str(executeCount[1]) + " instead of " + str(referenceCount[0]) + "+" + str(referenceCount[1]) + " executes")
            else:
                output("Executes  OK")

    # result: compare result files of visitems
    if CREATE_VISITEMS_REFERENCE:
        output("VisItems  WARNING: reference created")
    else:
        if not os.path.exists(currentRefDir):
            output("VisItems  ERROR: reference directory does not exist")
        else:
            errorFound = False
            for visitem in getFileListMatchingRegex(currentTmpDir, ".*\.covise"):
                if not os.path.exists(currentRefDir + "/" + visitem):
                    output("VisItems  ERROR: file " + visitem + " was created which is not in reference directory")
                    errorFound = True
            for visitem in getFileListMatchingRegex(currentRefDir, ".*\.covise"):
                if not os.path.exists(currentTmpDir + "/" + visitem):
                    output("VisItems  ERROR: file " + visitem + " was not created")
                    errorFound = True
                else:
                    diff = getFileDiff(currentRefDir + "/" + visitem, currentTmpDir + "/" + visitem)
                    if (visitem not in ignoreList) and (diff > 0):
                        if (diff > VISITEM_WARNING_MAX):
                            errorOrWarning = "ERROR"
                        else:
                            errorOrWarning = "WARNING"
                        output("VisItems  " + errorOrWarning + ": " + visitem + " differs at " + str(diff) + " byte(s)")
                        errorFound = True
            if not errorFound:
                output("VisItems  OK")

    # result: compare snapshots
    if CREATE_SNAPSHOT_REFERENCE:
        output("Snapshots WARNING: reference created")
    else:
        if not os.path.exists(currentRefDir):
            output("Snapshots ERROR: reference directory does not exist")
        else:
            errorFound = False
            for image in getFileListMatchingRegex(currentTmpDir, "snap.*_000.png"):
                if not os.path.exists(currentRefDir + "/" + image):
                    output("Snapshots ERROR: file " + image + " was created which is not in reference directory")
                    errorFound = True
            for image in getFileListMatchingRegex(currentRefDir, "snap.*_000.png"):
                if not os.path.exists(currentTmpDir + "/" + image):
                    output("Snapshots ERROR: file " + image + " was not created")
                    errorFound = True
                else:
                    diffImage = image.split(".")
                    diffImage[-2] = diffImage[-2] + "_DIFF"
                    diffImage = ".".join(diffImage)
                    diff = imageMagickCompare(currentRefDir + "/" + image, currentTmpDir + "/" + image, currentTmpDir + "/" + diffImage)
                    if (image not in ignoreList) and (diff != 0):
                        if (diff > SNAPSHOT_WARNING_MAX):
                            errorOrWarning = "ERROR"
                        else:
                            errorOrWarning = "WARNING"
                        output("Snapshots " + errorOrWarning + ": content of " + image + " differs at " + str(diff) + " pixel(s)")
                        errorFound = True
            if not errorFound:
                output("Snapshots OK")





######################################################################
#                           MAIN I (Test)
######################################################################

# enable some testing-functions via environment variables
os.environ["COVISE_EXECUTE_DEBUG"] = "1"
os.environ["VR_PREPARE_DEBUG_PRESENTATION_STEPS"] = "1"
os.environ["VR_PREPARE_DEBUG_QUIT"] = "1"

# make sure ../tmp/ and ../reference/ exist
safeMakeDir(TMP_DIR)
if CREATE_VISITEMS_REFERENCE or CREATE_EXECUTES_REFERENCE or CREATE_SNAPSHOT_REFERENCE:
    safeMakeDir(REFERENCE_DIR)

cleanUp()

# main loop
def recursion(subdir):
    for filename in getFileListMatchingRegexList(PROJECT_DIR + "/" + subdir, TEST_FILES_REGEX):
        test(subdir + filename)
    for dirname in getDirectoryList(PROJECT_DIR + "/" + subdir):
        recursion(subdir + dirname + "/")
recursion("")

# output
outfile.close()

######################################################################
#                           MAIN II (Result Output)
######################################################################

# summary
print("="*100)
print("Summary")
print("="*100)
entries = {}
entries["Testing"] = 0
entries["Console   OK"] = 0
entries["Executes  OK"] = 0
entries["VisItems  OK"] = 0
entries["Snapshots OK"] = 0
entries["Console   WARNING"] = 0
entries["Executes  WARNING"] = 0
entries["VisItems  WARNING"] = 0
entries["Snapshots WARNING"] = 0
entries["Console   ERROR"] = 0
entries["Executes  ERROR"] = 0
entries["VisItems  ERROR"] = 0
entries["Snapshots ERROR"] = 0
for line in fileinput.input(GLOBAL_LOG):
    for entry in entries.keys():
        if line.startswith(entry): entries[entry] = entries[entry] + 1
fileinput.close()
print("Projects tested: " + str(entries["Testing"]))
def printSummary(type):
    if (entries[type + "WARNING"] == 0) and (entries[type + "ERROR"] == 0):
        print(type + "OK")
    else:
        if (entries[type + "WARNING"] == 0):
            warning = ""
        else:
            warning = ", " + str(entries[type + "WARNING"]) + " warning(s)"
        print(type + str(entries[type + "ERROR"]) + " error(s)" + warning)
printSummary("Console   ")
printSummary("Executes  ")
printSummary("VisItems  ")
printSummary("Snapshots ")

# details
if (entries["Console   OK"] != entries["Testing"]) or \
   (entries["Executes  OK"] != entries["Testing"]) or \
   (entries["VisItems  OK"] != entries["Testing"]) or \
   (entries["Snapshots OK"] != entries["Testing"]):
    print
    print("="*100)
    print("Details")
    print("="*100)
    cache = []
    errorFound = False
    for line in fileinput.input(GLOBAL_LOG):
        if ("----------" in line) and (len(cache) > 0): # new test, print the old one first
            if errorFound:
                for c in cache:
                    print(c,)
            cache = []
            errorFound = False
        cache.append(line)
        if ("ERROR" in line) or ("WARNING" in line): errorFound = True
    fileinput.close()
    if errorFound:
        for c in cache:
            print(c,)

# active processes
print
print("="*100)
print("Active processes")
print("="*100)
for p in getProcesses():
    check = p.lower()
    if ("covise" in check) \
    or ("crb" in check) \
    or ("mapeditor" in check) \
    or ("scriptinterface" in check) \
    or ("cover" in check) \
    or ("vrprepare" in check) \
    or ("vr-prepare" in check) \
    or ("everest" in check):
        print(p)

