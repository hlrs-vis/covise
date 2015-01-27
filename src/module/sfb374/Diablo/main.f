C=======================================================================
      PROGRAM DIABLO
C=======================================================================
C     V A R I A B L E N                                       <<<<<<<<<<
C     ------------------------------------------------------------------
C   * COMMON-Block /MART/:
C     Martensitdaten, siehe SUBROUTINE RDMART
C
      COMMON /MART/ MS(5500),MF(5500),DTKRIT(5500),HMART(5500),
     1              HAUST(5500)
      REAL          MS, MF, DTKRIT
      INTEGER       HMART, HAUST
C
C     ------------------------------------------------------------------
C   * COMMON-Block /ZWISCH/:
C     ZwischenstufengefÅgedaten, siehe SUBROUTINE RDZWI
C
      COMMON /ZWISCH/ HZWI(5500,5500)
      INTEGER         HZWI
C
C     ------------------------------------------------------------------
C   * COMMON-Block /ZTA/:
C     NTAUZT   = Anzahl Zeit-StÅtzstellen des ZTA-Diagramms
C     NCCZTA   = Anzahl C-Konzentrations-StÅtzstellen des ZTA-Diagramms
C     TBEZZT   = Bezugstemperatur, ab der die Aufheizzeiten gelten
C     TAUZTA   = Zeit-StÅtzstellen
C     CCZTA    = C-Konzentrations-StÅtzstellen
C     AC3ZTA   = zugehîrige Ac3-Temperaturen = f(t,cC)
C
      COMMON /ZTA/     NTAUZT, NCCZTA, TBEZZT, 
     1                 TAUZTA(100), CCZTA(100), AC3ZTA(100,100)
      INTEGER          NTAUZT, NCCZTA
      DOUBLE PRECISION TBEZZT, TAUZTA, CCZTA, AC3ZTA
C
C     ------------------------------------------------------------------
C   * LOKALE Variablen:
C
C   - Parameterdatei
C     AFILE    = Name der Datei fÅr Austenitisierungs-(ZTA-)Diagramm
C     MFILE    = Name der Martensitdaten-Datei
C     ZFILE    = Name der ZwischenstufengefÅge-Datei
C     CCNODE   = durchschnittl. C-Konzentration des makroskop. Knotens
C                HIER: CCNODE = const. , Materialparameter !!!
C     LKORN    = char. LÑnge eines GefÅgekorns
C     FMIKRO   = Mikrostrukturfaktor, lMikro = fMikro*lKorn 
C     FMESH    = Anzahl Netzunterteilungen des mikroskopischen Netzes
C     NSTEPS   = Anzahl Zeitschritte bei der Diffusionsrechnung (Mikro)
C     CCMAX    = hîchster erlaubter C-Konzentrationswert (=cC,Perlit)
C     CCRESM   = Auflîsung der Martensitdaten-Arrays
C     CCRESZ   = Auflîsung der ZwischengefÅgedaten-Matrix bezÅglich cC
C     DTRESZ   = Auflîsung der ZwischengefÅgedaten-Matrix bezÅgl. dT/dt
C     TSCHM    = Schmelztemperatur
C     TX       = Abschrecktemperatur
C     EXEMOD   = ProgrammausfÅhrungsmodus
C     HDEF0    = Default-HÑrte ohne ErwÑrmung
C     HDEFWG   = Default-HÑrte mit ErwÑrmung, ohne vollstÑndige 
C                Austenitisierung (WeichglÅhhÑrte)
C     HDEFTS   = Default-HÑrte fÅr aufgeschmolzene Knoten
C     XLRES    = Zeilenanzahl bei Ausgabe von Diagrammen Excel-Dat.
C
      CHARACTER        AFILE*20, MFILE*20, ZFILE*20
      DOUBLE PRECISION CCNODE, LKORN, FMIKRO, CCMAX, TSCHM, TX
      INTEGER          FMESH, NSTEPS, CCRESM, CCRESZ, DTRESZ, EXEMOD
      INTEGER          HDEF0, HDEFWG, HDEFTS, XLRES, NDNMAX, MVNMAX
C
C   - interne Variablen
C     NDX      = Array mit x-Koordinaten der makroskopischen Knoten
C     NDY      = Array mit y-Koordinaten         -"-
C     NDZ      = Array mit z-Koordinaten         -"-
C     NDID     = Array mit FIDAP-Knotennummern der makrosk. Knoten
C     NDN      = Anzahl makroskopischer Knoten
C     NDNMAX   = maximale Anzahl makroskopischer Knoten
C     NDACT    = laufende Nummer des aktuellen makrosk. Knotens
C     NDHARD   = HÑrtewert                  -"-
C     MVTIME   = Array ZeitstÅtzstellen der Diffusionsrechnung
C     MVN      = Anzahl ZeitstÅtzstellen
C     MVNMAX   = maximale Anzahl ZeitstÅtzstellen
C     TIME1    = Startzeit fÅr Diffusion
C     TIME2    = Stopzeit fÅr Diffusion
C     TEMP1    = Startwert der Temperatur fÅr Diffusion
C     DTMAX    = max. Abschreckgeschw., aus (dT/dt)krit = f(cC)
C     ERRFLG   = Fehler-Code, = 0 wenn erfolgreich
C                Darf nur vom Hauptprogramm zurÅckgesetzt werden !!!
C     TEMPERATURES = Speicher pro Knoten die Temperaturkurven 
C                    -> kein langsamer Zugriff auf TVERLAUFE
C
      PARAMETER (NDNMAX=5500, MVNMAX=500)
      DOUBLE PRECISION NDX(NDNMAX), NDY(NDNMAX), NDZ(NDNMAX), DTMAX,
     1                 MVTIME(MVNMAX), TIME1, TIME2, TEMP1, RTIME,
     2                 TEMPERATURES(NDNMAX, MVNMAX)
      INTEGER          NDID(NDNMAX), NDN, NDACT, NDHARD, MVN, ERRFLG,
     1                 MCHECK, MRUN, MFAST, MLIN, MKUG
C
C     ------------------------------------------------------------------
C   * PARAMETER:
C
C   - Werte fÅr EXEMOD
C
      PARAMETER (MCHECK=1, MRUN=2, MFAST=3, MLIN=4, MKUG=5)
C
C     ------------------------------------------------------------------
C     H A U P T P R O G R A M M                               <<<<<<<<<<
C     ------------------------------------------------------------------
      PRINT*,'--------------- PROGRAMM DIABLO ---------------'
      PRINT*
      ERRFLG = 0
C
C     ------------------------------------------------------------------
C     Parameterdatei einlesen
C
      PRINT*
      CALL RDPARM(AFILE, MFILE, ZFILE, CCNODE, LKORN, FMIKRO, 
     1            FMESH, NSTEPS, CCMAX, CCRESM, CCRESZ, DTRESZ, 
     2            TSCHM, TX, EXEMOD, HDEF0, HDEFWG, HDEFTS, 
     3            XLRES, ERRFLG)
C                                                      >>> E R R O R <<<
      IF (ERRFLG.NE.0) THEN
        PRINT*
        PRINT*,'Programmabbruch:'
        PRINT*,'Fehler in Parameterdatei !!!'
        GOTO 10000
      ENDIF
C
C     ------------------------------------------------------------------
C     Martensitdaten einlesen
C     ==> COMMON-Block /MART/ belegen
C         DTMAX bestimmen (maximale kritische Abschreckgeschwindigkeit)
C
      PRINT*
      CALL RDMART (MFILE, CCMAX, CCRESM, TSCHM, DTMAX, ERRFLG)
C
C     ------------------------------------------------------------------
C     ZwischengefÅgedaten einlesen
C     ==> COMMON-Block /ZWI/ belegen
C
      PRINT*
      CALL RDZW (ZFILE, CCMAX, CCRESZ, DTMAX, DTRESZ, ERRFLG)
C
C     ------------------------------------------------------------------
C     ZTA-Diagramm einlesen
C     ==> COMMON-Block /ZTA/ belegen
C
      PRINT*
      CALL RDZTA (AFILE, TSCHM, ERRFLG)
C
C     ------------------------------------------------------------------
C     Diffusionskoeffizienten einlesen
C     ( ==> COMMON-Block /DKOEFF/ belegen ) nicht installiert !!!
C
C     PRINT*
      CALL RDDK (ERRFLG)
C
C     ------------------------------------------------------------------
C     Fehler beim Einlesen der Materialdateien ?
C                                                      >>> E R R O R <<<
      IF (ERRFLG.NE.0) THEN
        PRINT*
        PRINT*,'Programmabbruch:'
        PRINT*,'Fehler in Materialdatei(en) !!!'
        GOTO 10000
      ELSE
        PRINT*
        PRINT*,'Eingabedaten OK'
      ENDIF
C
C     ------------------------------------------------------------------
C     Programm entsprechend gewÑhltem Modus ausfÅhren
C
C     ------------------------------------------------------------------
C     CHECK-Modus
C
      IF (EXEMOD.EQ.MCHECK) THEN
C       ----------------------------------------------------------------
C       Materialdaten-Arrays ausgeben
C       ==> COMMON-Blocks 
C           /MART/ , /ZWISCH/ 
C           in Excel-Dateien speichern
C
	PRINT*
	PRINT*,'CHECK-Modus: Ausgabe in Excel-Dateien:'
        CALL WRXCEL (CCMAX, DTMAX, CCRESM, CCRESZ, DTRESZ, XLRES, 
     1               ERRFLG)
C
C     ------------------------------------------------------------------
C     RUN-/FAST-/1DIMLIN-/1DIMKUG-Modus
C
      ELSEIF ((EXEMOD.EQ.MRUN).OR.(EXEMOD.EQ.MFAST).OR.
     1        (EXEMOD.EQ.MLIN).OR.(EXEMOD.EQ.MKUG)) THEN
        PRINT*
	PRINT*,'>>> BERECHNUNG <<<'
	PRINT*
        OPEN (1,FILE='HAERTE')
	WRITE (1,'(a5,a10,2a15,a12)',ERR=15) 'ND','X','Y','Z','HAERTE'
	goto 16
C     ------------------------------------------------------------------
C     Datei-Schreibfehler
C                                                      >>> E R R O R <<<
15        PRINT*
	  PRINT*,'Fehler beim Schreiben in Ausgabedatei HAERTE'
C
C       ----------------------------------------------------------------
16      print *
C       Einlesen, welche Knoten betrachtet werden sollen, 
C       "temp.FIOUT" aufbereiten.
C
	CALL PREP (NDID, NDX, NDY, NDZ, NDN, NDNMAX,
     1             MVTIME, MVN, MVNMAX, TEMPERATURES, ERRFLG)
        IF (ERRFLG.NE.0) THEN
C       ----------------------------------------------------------------
C                                                      >>> E R R O R <<<
	   PRINT*
	   PRINT*,'Programmabbruch:'
	   PRINT*,'Fehler bei Aufbereitung der Makro-Ergebnisdatei !!!'
	   GOTO 10000
        ENDIF
C       ----------------------------------------------------------------
C       Berechnung fÅr alle relevanten Knoten
C    
        DO 2000, NDACT=1, NDN
          PRINT*
          PRINT*,'--------------------------'
          PRINT 100, NDACT, NDN
100       FORMAT('>>> Knoten ',i4,' von ',i4)
          PRINT*
C         --------------------------------------------------------------
C         "T_VERLAUF" erstellen
C    
          CALL WRTVER (NDACT, NDN, MVTIME, MVN, CCNODE, CCMAX,
     1                 CCRESM, TEMP1, TIME1, TSCHM, TIME2, NDNMAX, 
     2                 MVNMAX, TEMPERATURES, ERRFLG)
C         --------------------------------------------------------------
C         ERROR-Code von WRTVER auswerten
C    
C         Fehlerfrei: normale Rechnung starten
          IF (ERRFLG.EQ.0) THEN
            GOTO 1100
C         nicht aufgeheizt: DefaulthÑrte
          ELSEIF ((ERRFLG.GT.0).AND.(ERRFLG.LE.1)) THEN
            NDHARD = HDEF0
            GOTO 1400
C         nicht vollst. austenitisiert: WeichglÅhhÑrte
          ELSEIF ((ERRFLG.GT.1).AND.(ERRFLG.LE.3)) THEN
            NDHARD = HDEFWG
            GOTO 1400
C         aufgeschmolzen: DefaulthÑrte, Schmelze
          ELSEIF ((ERRFLG.GT.3).AND.(ERRFLG.LE.4)) THEN
            NDHARD = HDEFTS
            GOTO 1400
          ELSE
C       ----------------------------------------------------------------
C                                                      >>> E R R O R <<<
	    PRINT*
	    PRINT*,'Fehler beim Erstellen der Temperaturkurve !!!'
	    PRINT*,'HÑrte wird als DefaulthÑrte angenommen !!!'
	    GOTO 1400
          ENDIF
C         --------------------------------------------------------------
C         Nur RUN-Mode: Diffusionsrechnung vorbereiten und starten
C 
1100      IF (EXEMOD.EQ.MRUN) THEN
C           ------------------------------------------------------------
C           "FDREAD.diff" erstellen
C 
	    CALL WRFIIN (LKORN, FMIKRO, FMESH, NSTEPS, TIME1, TIME2,
     1                   TEMP1, ERRFLG)
            IF (ERRFLG.NE.0) THEN
C       ----------------------------------------------------------------
C                                                      >>> E R R O R <<<
	       PRINT*
	       PRINT*,'Fehler beim Erstellen der FIDAP-Eingabedatei !!!'
	       PRINT*,'HÑrte wird als DefaulthÑrte angenommen !!!'
	       GOTO 1400
            ENDIF
C           ------------------------------------------------------------
C           FIDAP starten
C
            CALL SYSTEM ('CALC.DIFF')
          ENDIF
C
C    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
C         --------------------------------------------------------------
C         Nur 1DIM-DIFF.-Mode:Diffusionsrechnung vorbereiten und starten
C
1200      IF ((EXEMOD.EQ.MLIN).or.(EXEMOD.EQ.MKUG)) THEN
C
	     call dif1dim (TEMPERATURES, MVN, NDACT, TIME1, TIME2, 
     1                     CCNODE, CCMAX, LKORN, FMESH, NSTEPS, 
     2                     TEMP1, EXEMOD, ERRFLG)
C
	     IF (ERRFLG.NE.0) THEN
C       ----------------------------------------------------------------
C                                                      >>> E R R O R <<<
		PRINT*
		PRINT*,'Fehler bei der 1dim-Diff.-Rechnung !!!'
		PRINT*,'HÑrte wird als DefaulthÑrte angenommen !!!'
		GOTO 1400
	     ENDIF
	  ENDIF
C    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
C         --------------------------------------------------------------
C         HÑrtewert bestimmen
C
          CALL HARD (NDHARD, TSCHM, TX, CCMAX, CCRESM, CCRESZ, DTRESZ, 
     1               DTMAX, EXEMOD, CCNODE, FMESH, ERRFLG)
          IF (ERRFLG.NE.0) THEN
C       ----------------------------------------------------------------
C                                                      >>> E R R O R <<<
	     PRINT*
	     PRINT*,'Fehler bei HÑrtewert-Berechnung !!!'
	     PRINT*,'HÑrte wird als DefaulthÑrte angenommen !!!'
	     GOTO 1400
          ENDIF
C         --------------------------------------------------------------
C         Ergebnisausgabe
C
1400      PRINT*
          PRINT 1410, NDHARD
1410      FORMAT('>>> HÑrtewert = ',i6)
	  WRITE (1,'(i5,3e15.5,i6)',ERR=1500)
     1           NDID(NDACT), NDX(NDACT), NDY(NDACT), NDZ(NDACT), NDHARD
C
C     ------------------------------------------------------------------
C     Datei-Schreibfehler
C                                                      >>> E R R O R <<<
	  goto 1600
1500      PRINT*
	  PRINT*,'Fehler beim Schreiben in Ausgabedatei HAERTE'
C
1600      ERRFLG = 0
2000    CONTINUE
        GOTO 10000 
C     ------------------------------------------------------------------
C     Falscher Modus
C                                                      >>> E R R O R <<<
      ELSE
        PRINT*
        PRINT*,'Programmabbruch:'
        PRINT*,'AusfÅhrungsmodus falsch angegeben'
	PRINT*,'CHECK, RUNDIFF, RUNFAST, 1DIMLIN, 1DIMKUG'
      ENDIF
      GOTO 10000
C
C     ------------------------------------------------------------------
C     H A U P T P R O G R A M M                --->   E N D E <<<<<<<<<<
C     ------------------------------------------------------------------
C
10000 PRINT*
      RTIME= mclock()
c     'mclock' liefert die Rechenzeit in 1/100 Sekunden !
      RTIME=RTIME*0.01
      print *,'Benîtigte Rechenzeit : ',RTIME
      PRINT*,'Programm beendet'
      CLOSE(1,STATUS='KEEP')
      END
C

