@ECHO OFF

REM (C) Copyright 2009 VISENSO GmbH
REM author: Harry Trautmann
REM *******************************************
REM Sets the local paths of applications needed
REM to either compile or run COVISE that are 
REM outside the EXTERNLIBS directory or differ
REM of the default path
REM *******************************************

REM *******************************************
REM let user know what will be set
ECHO Adapting environment for computer
ECHO ...
REM todo: SPECIFY-COMPUTERNAME-HERE, e. g. like 
REM ECHO     o2.vircinity
ECHO ...
REM *******************************************

REM entries could be
REM add path to Visual Studio´s vcvarsall.bat
REM SET PATH=%PATH%;C:\Programme\Microsoft Visual Studio 8\VC
REM SET EXTERNLIBS=C:\vista
REM SET "INNOSETUPHOME=C:\Programme\Inno Setup 5"
REM SET "PYTHONHOME=%EXTERNLIBS%\Python-2.5.4"
REM SET PYTHONVERSION=25
REM SET "QTDIR=%EXTERNLIBS%\Qt_4.4.3"
REM SET COCONFIG=config.xml