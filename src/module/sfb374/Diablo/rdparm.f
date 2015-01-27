C=======================================================================
C     Parameterdatei einlesen
C     ------------------------------------------------------------------
C
C     ------------------------------------------------------------------
      SUBROUTINE RDPARM(AFILE, MFILE, ZFILE, CCNODE, LKORN, FMIKRO, 
     1                  FMESH, NSTEPS, CCMAX, CCRESM, CCRESZ, DTRESZ, 
     2                  TSCHM, TX, EXEMOD, HDEF0, HDEFWG, HDEFTS, 
     3                  XLRES, ERRFLG)
C
C     ------------------------------------------------------------------
C   * Aufruf-Parameter:
C o   AFILE    = Name der Datei fr Austenitisierungs-(ZTA-)Diagramm
C o   MFILE    = Name der Martensitdaten-Datei
C o   ZFILE    = Name der Zwischenstufengefge-Datei
C o   CCNODE   = durchschnittl. C-Konzentration des makroskop. Knotens
C                HIER: CCNODE = const. , Materialparameter !!!
C o   LKORN    = char. L„nge eines Gefgekorns
C o   FMIKRO   = Mikrostrukturfaktor, lMikro = fMikro*lKorn 
C o   FMESH    = Anzahl Netzunterteilungen des mikroskopischen Netzes
C o   NSTEPS   = Anzahl Zeitschritte bei der Diffusionsrechnung (Mikro)
C o   CCMAX    = h”chster erlaubter C-Konzentrationswert (=cC,Perlit)
C o   CCRESM   = Aufl”sung der Martensitdaten-Arrays
C o   CCRESZ   = Aufl”sung der Zwischengefgedaten-Matrix bezglich cC
C o   DTRESZ   = Aufl”sung der Zwischengefgedaten-Matrix bezgl. dT/dt
C o   TSCHM    = Schmelztemperatur
C o   TX       = Abschrecktemperatur
C o   EXEMOD   = Programmausfhrungsmodus
C o   HDEF0    = Default-H„rte ohne Erw„rmung
C o   HDEFWG   = Default-H„rte mit Erw„rmung, ohne vollst„ndige 
C                Austenitisierung (Weichglhh„rte)
C o   HDEFTS   = Default-H„rte fr aufgeschmolzene Knoten
C o   XLRES    = Zeilenanzahl bei Ausgabe von Diagrammen Excel-Dat.
C o   ERRFLG   = Fehler-Code, = 1 wenn Fehler, sonst unver„ndert
C
      CHARACTER        AFILE*20, MFILE*20, ZFILE*20
      DOUBLE PRECISION CCNODE, LKORN, FMIKRO, CCMAX, TSCHM, TX
      INTEGER          FMESH, NSTEPS, CCRESM, CCRESZ, DTRESZ, EXEMOD,
     1                 MCHECK, MRUN, MFAST, MLIN, MKUG, IOERROR
      INTEGER          HDEF0, HDEFWG, HDEFTS, XLRES, ERRFLG
C
C     ------------------------------------------------------------------
C   * interne Variablen:
C     MODSTR   = String zum Speichern des angegebenen Modus
C     STRG1,2  = Hilfs-Strings zum Einlesen
      CHARACTER STRG1*1, STRG2*10, MODSTR*20
C
C     ------------------------------------------------------------------
C   * PARAMETER:
C
C   - Werte fr EXEMOD
C
      PARAMETER (MCHECK=1, MRUN=2, MFAST=3, MLIN=4, MKUG=5)
C
C     ------------------------------------------------------------------
C                                                      >>> P R I N T <<<
      PRINT*,'Parameterdatei einlesen'
C
C     ------------------------------------------------------------------
C     Datei "Parameter" ”ffnen
C
      OPEN (10,FILE='Parameter',IOSTAT=IOERROR)
C     ------------------------------------------------------------------
C     Parameterdatei vollst„ndig einlesen
C
10    READ (10,'(a1,29x,a10)',ERR=20,END=20) STRG1, STRG2
      IF (STRG1.EQ.'C') THEN
	GOTO 10
      END IF
      BACKSPACE 10
      IF     (STRG2.EQ.'ZTADiagr  ') THEN
	READ (10,'(52x,a20)',ERR=20,END=20) AFILE
      ELSEIF (STRG2.EQ.'Martensit ') THEN
	READ (10,'(52x,a20)',ERR=20,END=20) MFILE
      ELSEIF (STRG2.EQ.'Zwischgef ') THEN
	READ (10,'(52x,a20)',ERR=20,END=20) ZFILE
      ELSEIF (STRG2.EQ.'CGehalt   ') THEN
	READ (10,'(52x,f20.3)',ERR=20,END=20) CCNODE
      ELSEIF (STRG2.EQ.'lKorn     ') THEN
	READ (10,'(52x,f20.3)',ERR=20,END=20) LKORN
      ELSEIF (STRG2.EQ.'fMikro    ') THEN
	READ (10,'(52x,f20.3)',ERR=20,END=20) FMIKRO
      ELSEIF (STRG2.EQ.'fMesh     ') THEN
	READ (10,'(52x,i20)',ERR=20,END=20) FMESH
      ELSEIF (STRG2.EQ.'nSteps    ') THEN
	READ (10,'(52x,i20)',ERR=20,END=20) NSTEPS
      ELSEIF (STRG2.EQ.'cCMax     ') THEN
	READ (10,'(52x,f20.3)',ERR=20,END=20) CCMAX
      ELSEIF (STRG2.EQ.'cCResMart ') THEN
	READ (10,'(52x,i20)',ERR=20,END=20) CCRESM
      ELSEIF (STRG2.EQ.'cCResZwi  ') THEN
	READ (10,'(52x,i20)',ERR=20,END=20) CCRESZ
      ELSEIF (STRG2.EQ.'dTResZwi  ') THEN
	READ (10,'(52x,i20)',ERR=20,END=20) DTRESZ
      ELSEIF (STRG2.EQ.'TSchmelz  ') THEN
	READ (10,'(52x,f20.3)',ERR=20,END=20) TSCHM
      ELSEIF (STRG2.EQ.'TAbschrck ') THEN
	READ (10,'(52x,f20.3)',ERR=20,END=20) TX
      ELSEIF (STRG2.EQ.'ExecModus ') THEN
	READ (10,'(52x,a20)',ERR=20,END=20) MODSTR
      ELSEIF (STRG2.EQ.'HDefault  ') THEN
	READ (10,'(52x,i20)',ERR=20,END=20) HDEF0
      ELSEIF (STRG2.EQ.'HWeichgl  ') THEN
	READ (10,'(52x,i20)',ERR=20,END=20) HDEFWG
      ELSEIF (STRG2.EQ.'HSchmelz  ') THEN
	READ (10,'(52x,i20)',ERR=20,END=20) HDEFTS
      ELSEIF (STRG2.EQ.'ExcelRes  ') THEN
	READ (10,'(52x,i20)',ERR=20,END=20) XLRES
      ELSE
	READ (10,'(a1)',ERR=20,END=20) STRG1
      END IF
      GOTO 10
20    CLOSE(10,STATUS='KEEP')
C
C     ------------------------------------------------------------------
C     Eingelesene Werte auswerten, berprfen
C
      IF     (MODSTR.EQ.'CHECK') THEN
	EXEMOD = MCHECK
      ELSEIF (MODSTR.EQ.'RUNDIFF') THEN
	EXEMOD = MRUN
      ELSEIF (MODSTR.EQ.'RUNFAST') THEN
	EXEMOD = MFAST
      ELSEIF (MODSTR.EQ.'1DIMLIN') THEN
	EXEMOD = MLIN
      ELSEIF (MODSTR.EQ.'1DIMKUG') THEN
	EXEMOD = MKUG
      ELSE
	EXEMOD = 0
      END IF
C
      IF (CCNODE.LE.0) THEN
	PRINT*,'C-Gehalt kleiner oder gleich 0% !!!'
	ERRFLG = 1
      END IF
      IF (LKORN.LE.0) THEN
	PRINT*,'Korngr”áe kleiner oder gleich 0 !!!'
	ERRFLG = 1
      END IF
      IF (FMIKRO.LE.0) THEN
	PRINT*,'Mikrostrukturfaktor kleiner oder gleich 0 !!!'
	ERRFLG = 1
      END IF
      IF ((FMESH.LE.0).OR.(FMESH.GT.80)) THEN
	PRINT*,'Meshfaktor auáerhalb Wertebereich (1...80) !!!'
	ERRFLG = 1
      END IF
      IF (NSTEPS.LE.0) THEN
	PRINT*,'Anzahl Zeitschritte kleiner oder gleich 0 !!!'
	ERRFLG = 1
      END IF
      IF (CCMAX.LE.0) THEN
	PRINT*,'Max. C-Gehalt kleiner oder gleich 0% !!!'
	ERRFLG = 1
      END IF
      IF ((CCRESM.LT.10).OR.(CCRESM.GT.1000)) THEN
	PRINT*,'Aufl”sung fr Martensitdaten-Array falsch !!!'
	ERRFLG = 1
      END IF
      IF ((CCRESZ.LT.10).OR.(CCRESZ.GT.1000)) THEN
	PRINT*,'C-Aufl”sung fr Zwischengefgedaten-Array falsch !!!'
	ERRFLG = 1
      END IF
      IF ((DTRESZ.LT.10).OR.(DTRESZ.GT.1000)) THEN
	PRINT*,'dT/dt-Aufl”sung fr Zwischengef.daten-Array falsch !!!'
	ERRFLG = 1
      END IF
      IF ((TSCHM.LT.1).OR.(TSCHM.GT.3000)) THEN
	PRINT*,'Schmelztemperatur kleiner 1K oder gr”áer 3000K !!!'
	ERRFLG = 1
      END IF
      IF ((TX.LT.1).OR.(TX.GT.TSCHM)) THEN
	PRINT*,'Abschrecktemp. kleiner 1K oder gr”áer Tschmelz !!!'
	ERRFLG = 1
      END IF
      IF ((XLRES.LE.1).OR.(XLRES.GT.100)) THEN
	XLRES = 25
      END IF
C
C     ------------------------------------------------------------------
      IF (ERRFLG.NE.0) THEN
C                                                      >>> E R R O R <<<
	PRINT*,'Fehler in der Parameterdatei !!!'
	RETURN
      END IF
C
C     ------------------------------------------------------------------
C                                                      >>> P R I N T <<<
      PRINT*,'erfolgreich beendet.'
      RETURN
      END
C
C=======================================================================
