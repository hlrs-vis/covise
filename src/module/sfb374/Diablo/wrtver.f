C=======================================================================
C     Datei "T_VERLAUF" erstellen
C     ------------------------------------------------------------------
C     Bereitet die Temperaturkurve fr den aktuellen makroskopischen
C     Knoten aus den Angaben in "T_VERLAEUFE" auf.
C     ------------------------------------------------------------------
      SUBROUTINE WRTVER (NODENO, NNODES, TAU, NTAU, CCNODE, CCMAX,
     1                   CCRESM, TAC3, TAUAC3, TAUA3X, TSCHM, TAUMS, 
     2                   NOMAX, NTAUMAX, TEMPARR, ERRFLG) 
C     ------------------------------------------------------------------
C   * COMMON-Block /MART/:
C i   MS       = Martensitstarttemperatur
C i   MF       = Martensitfinishtemperatur
C i   DTKRIT   = kritische Abschreckgeschwindigkeit
C i   HMART    = H„rtewert von 100%-igem Martensit
C i   HAUST    = Austenith„rte
C
      COMMON /MART/ MS(5500),MF(5500),DTKRIT(5500),HMART(5500),
     1              HAUST(5500)
      REAL          MS, MF, DTKRIT
      INTEGER       HMART, HAUST
C
C     ------------------------------------------------------------------
C   * COMMON-Block /ZTA/:
C i   NTAUZT   = Anzahl Zeit-Sttzstellen des ZTA-Diagramms
C i   NCCZTA   = Anzahl C-Konzentrations-Sttzstellen des ZTA-Diagramms
C i   TBEZZT   = Bezugstemperatur, ab der die Aufheizzeiten gelten
C i   TAUZTA   = Zeit-Sttzstellen
C i   CCZTA    = C-Konzentrations-Sttzstellen
C i   AC3ZTA   = zugeh”rige Ac3-Temperaturen = f(t,cC)
C
      COMMON /ZTA/     NTAUZT, NCCZTA, TBEZZT, 
     1                 TAUZTA(100), CCZTA(100), AC3ZTA(100,100)
      INTEGER          NTAUZT, NCCZTA
      DOUBLE PRECISION TBEZZT, TAUZTA, CCZTA, AC3ZTA
C
C     ------------------------------------------------------------------
C   * AUFRUF-Parameter:
C i   NODENO   = laufende Nummer (NICHT FIDAP-Nummer) des Knotens
C i   NNODES   = Anzahl relevanter Knoten = Anzahl Zeilen pro Datenblock
C i   TAU      = Array mit zu den Datenbl”cken geh”rigen MOVIE-Zeiten
C i   NTAU     = Anzahl MOVIE-Zeiten = Anzahl Datenbl”cke
C i   CCNODE   = Kohlenstoffkonzentration des Knotens
C i   CCMAX    = maximaler C-Konzentrationswert
C i   CCRESM   = Aufl”sung der Martensitdaten-Arrays
C o   TAC3     = Ac3-Temperatur im Schnittpkt mit Austenitisierungskurve
C o   TAUAC3   = Zeitpunkt, an dem Ac3-Temperatur TAC3 erreicht wird
C o   TAUMS    = Zeitpunkt, an dem Martensitstarttemp. erreicht wird
C i   TSCHM    = Schmelztemperatur
C i   NOMAX    = Maximale Kontenanzahl
C i   NTAUMAX  = MAximale Anzahl der Temperaturstuetzstellen
C i   TEMPARR  = Enthaelt die Temperaturkurven pro Knoten
C o   ERRFLG   = Fehler-Code:
C                --- keine H„rterechnung n”tig/m”glich ---
C                =  1  keine Aufheizung um mehr als 10K
C                =  2  Gefge wird nicht vollst„ndig austenitisiert
C                =  3  Kein Schnittp. mit ZTA-Diag. nach 100 Iterationen
C                =  4  Schmelztemperatur berschritten 
C                --- schwerer Fehler ---
C                =  5  keine Abkhlung auf Martensitstarttemperatur
C                =  6  erste Temp. gr”áer (Ac3,unendl.-10K)
C                =  7  C-Konzentration des Knotens auáerhalb ZTA-Bereich
C                =  8  Martensitstarttemp. > Ac3,unendlich
C                =  9  Temperaturwert kleiner 1K
C                = 10  Datei-Lese- oder Schreibfehler
C                sonst unver„ndert
C               
C      DOUBLE PRECISION TAU(NTAU), CCNODE, CCMAX, TAC3, TSCHM,
      INTEGER          NODENO, NNODES, NTAU, CCRESM, ERRFLG,
     1                 NOMAX, NTAUMAX 
C      CHARACTER        PATH*120
      DOUBLE PRECISION TAU(500), CCNODE, CCMAX, TAC3, TSCHM, TAUA3X
     1                 TAUAC3, TAUMS, TEMPARR(5500, 500)
C
C     ------------------------------------------------------------------
C   * Interne Variablen:
C     T        = eingelesene Temperaturwerte
C     TAUA3X   = Zeitpunkt, an dem Ac3,unendlich erreicht wird
C     TAC3X    = Temperatur Ac3,unendlich (=letzte Temp. im ZTA-Diagr.)
C     T1,T2    = Temperaturwerte zur Iteration
C     DT       = mittlere Aufheizgeschwindigkeit
C     TAUHLP   = Hilfsvariable fr Zeit zum Ablesen aus /ZTA/
C     THLP     = Hilfsvariable fr Temperatur zur Iteration
C     CCNRMM   = normierte C-Konzentration fr Martensitdaten
C     I,J      = Z„hlvariable
C     ISTART   = Startsttzstelle, kontextabh„ngig
C     STRG     = Hilfs-String zum Einlesen
C
C      DOUBLE PRECISION T(NTAU), TAC3X, T1, T2, DT, TAUHLP, THLP
      DOUBLE PRECISION T(500), TAUA3X, TAC3X, T1, T2, DT, TAUHLP, THLP,
     1                 TAU1, TAU2, EGON, TEMP
      CHARACTER STRG*1
      INTEGER   CCNRMM, I, J, ISTART
C
C     ------------------------------------------------------------------
C                                                      >>> P R I N T <<<
C      PRINT*,'Temperaturkurve wird aufbereitet'
C
C     ------------------------------------------------------------------
C     Dateien ”ffnen
C
C      OPEN (10,FILE='T_VERLAEUFE')
C      OPEN (11,FILE=PATH)
C
C     ------------------------------------------------------------------
C     Temperaturwerte aus "T_VERLAEUFE" einlesen
C
C      DO 10 I = 1, (NODENO-1)
C	READ (10,'(a1)',ERR=50,END=50) STRG
C10    CONTINUE
C      DO 30 I = 1, NTAU
C	READ (10,'(49x,e17.8)',ERR=50,END=50) T(I)
C	DO 20 J = 1, (NNODES-1)
C	  READ (10,'(a1)',ERR=100,END=100) STRG
C20      CONTINUE
C30    CONTINUE
      GOTO 100
C
C     ------------------------------------------------------------------
C     normierte C-Konzentration CCNRMM bestimmen. 
C     Prfen, ob C-Konzentration des Knotens innerhalb ZTA-Wertebereich. 
C     Ac3,unendlich (fr letzten t-Wert im ZTA-Diagramm) bestimmen.
C
100   DO 105 I = 1, NTAU
        T(I) = TEMPARR(NODENO, I)
105   CONTINUE
      CCNRMM = 1+CCNODE*(CCRESM-1)/CCMAX
      IF ((CCNODE.LT.CCZTA(1)).OR.(CCNODE.GT.CCZTA(NCCZTA))) THEN
	GOTO 370
      ENDIF
      CALL IPOL2P ( TAUZTA, CCZTA, AC3ZTA, TAUZTA(NTAUZT), CCNODE,
     1              TAC3X, NTAUZT, NCCZTA)
C
C     ==================================================================
C     Eingelesene Temperaturkurve auswerten, charakteristische Zeiten
C     bestimmen.
C
C     ------------------------------------------------------------------
C     Prfen, ob erster Temperaturwert bereits zu hoch (max.TAC3X-10K)
C
110   IF (T(1).GT.(TAC3X-10)) THEN
	GOTO 360
      ENDIF
C     ------------------------------------------------------------------
C     Prfen, ob Ac3,unendl. kleiner als Martensitstarttemperatur
C     => Abkhlkurve kann nicht berechnet werden.
C
120   IF (TAC3X.LE.MS(CCNRMM)) THEN
	GOTO 380
      ENDIF
C     ------------------------------------------------------------------
C     Schnittpunkt mit ZTA-Diagramm bestimmen:
C     1. Letzte Sttzstelle ISTART suchen, nach der die Anfangstemp. 
C        T(1) um mehr als 10K berschritten wird. Ab T(ISTART) wird die
C        Aufheizkurve ausgewertet
C      ! Das ZTA-Diagramm kann bei ISTART noch nicht geschnitten worden
C        sein (T(1) kleiner (Ac3,unendl.-10K) ,s.o.), deswegen ist eine 
C        entsprechende Fehlerabfrage nicht n”tig.
C     2. Fr jede folgende Sttzstelle I wird die durchschnittliche Auf-
C        heizgeschw. DT ab T(ISTART) bestimmt. Mit DT wird die Zeit 
C        TAUHLP bestimmt, die ben”tigt worden w„re, um T(I) bei konti-
C        nuierlicher Aufheizung (dT/dt=const.=DT) von der ZTA-Bezugs-
C        temperatur TBEZZT an zu erreichen. Aus dem ZTA-Diagramm wird 
C        die zu TAU2 geh”rige Ac3-Temperatur TAC3 bestimmt. Ist TAC3 
C        kleiner als T(I), so bedeutet dies, daá zwischen dieser und der 
C        vorhergehenden Sttzstelle das ZTA-Diagramm geschnitten wurde.
C     3. Ist das Sttzstellenpaar (I-1),(I) gefunden, zwischen dem das
C        ZTA-Diagramm geschnitten wurde, so muá der tats„chl. Schnitt-
C        punkt iterativ bestimmt werden. Dies geschieht durch Intervall-
C        schachtelung. Die Wertepaare TAU(I-1),T(I-1) und TAU(I),T(I)
C        werden als erste Unter- und Obergrenze TAU1,T1 und TAU2,T2
C        gew„hlt.
C        Intervallschachtelung:
C        Das Intervall wird halbiert, indem TAUAC3,THLP als
C        TAUAC3=(TAU1+TAU2)/2 und THLP=(T1+T2)/2 bestimmt werden. Ist
C        die zugeh”rige Ac3-Temperatur (ZTA-Diagramm) gr”áer als 
C        THLP+1 (TOLERANZ !), so wird das Wertepaar TAUAC3,THLP zur
C        Untergrenze TAU1,T1 des Intervalls. Ist sie kleiner als 
C        THLP-1, so wird TAUAC3,THLP zur Obergrenze TAU2,T2. 
C        Andernfalls gilt der Schnittpunkt als bestimmt.
C        Werden mehr als 100 Iterationsschritte ben”tigt, so gilt der 
C        Schnittpunkt als nicht bestimmbar !
C
C     ---1---
      DO 130 I = 1, NTAU-1
        TEMP = T(I)
	IF (T(I).GE.TSCHM) THEN
	  GOTO 340
	ELSEIF (T(I).LT.1) THEN
	  GOTO 390
	ENDIF
	IF (T(I+1).GT.(T(1)+10)) THEN
	  ISTART = I
	  GOTO 140
	ENDIF
130   CONTINUE
      GOTO 310
C
C     ---2---
140   DO 160 I = (ISTART+1), NTAU
	IF (T(I).GE.TSCHM) THEN
	  GOTO 340
	ELSEIF (T(I).LT.1) THEN
	  GOTO 390
	ENDIF
	DT = (T(I)-T(ISTART))/(TAU(I)-TAU(ISTART))
	TAUHLP = (T(I)-TBEZZT)/DT
	CALL IPOL2P ( TAUZTA, CCZTA, AC3ZTA, TAUHLP, CCNODE,
     1                TAC3, NTAUZT, NCCZTA)
C
C       ---3---
	IF (TAC3.LT.T(I)) THEN
	  TAU1 = TAU(I-1)
	  T1   = T(I-1)
	  TAU2 = TAU(I)
	  T2   = T(I)
	  DO 150 J=1, 100
	    EGON = (TAU1+TAU2)/2
C                                                      >>> D E B U G <<<
C         PRINT*,'t1,t2:',TAU1,TAU2
C         PRINT*,'t:',EGON
	    THLP   = (T1+T2)/2
	    DT = (THLP-T(ISTART))/(EGON-TAU(ISTART))
	    TAUHLP = (THLP-TBEZZT)/DT
	    CALL IPOL2P ( TAUZTA, CCZTA, AC3ZTA, TAUHLP, CCNODE,
     1                    TAC3, NTAUZT, NCCZTA)
C                                                      >>> D E B U G <<<
C         PRINT*,'t,Ac3,T:',EGON,TAC3,THLP
	    IF     (TAC3.GT.(THLP+1)) THEN
	      TAU1 = EGON
	      T1   = THLP
	    ELSEIF (TAC3.LT.(THLP-1)) THEN
	      TAU2 = EGON
	      T2   = THLP
	    ELSE
	      TAC3   = THLP
	      TAUAC3 = EGON
C                                                      >>> D E B U G <<<
C         PRINT*,'WRTVER: T=',TAC3
	      GOTO 200
	    ENDIF
150       CONTINUE
	  GOTO 330
	ENDIF
C       -------
160   CONTINUE
      GOTO 320
C     -------
C
C     ------------------------------------------------------------------
C     Schnittpunkt mit TAC3X bestimmen:
C
200   ISTART = I
      DO 210 I = ISTART, NTAU-1
	IF (T(I).GE.TSCHM) THEN
	  GOTO 340
	ELSEIF (T(I).LT.1) THEN
	  GOTO 390
	ELSEIF ((T(I).GT.TAC3X).AND.(T(I+1).LE.TAC3X)) THEN
	  TAUA3X = TAU(I)+(TAC3X-T(I))/(T(I+1)-T(I))*(TAU(I+1)-TAU(I))
	  GOTO 220
	ENDIF
210   CONTINUE
      GOTO 350
C
C     ------------------------------------------------------------------
C     Schnittpunkt mit Ms bestimmen:
C
220   ISTART = I
      DO 230 I = ISTART, NTAU-1
	IF (T(I).GE.TSCHM) THEN
	  GOTO 340
	ELSEIF (T(I).LT.1) THEN
	  GOTO 390
	ENDIF
	IF ((T(I).GT.MS(CCNRMM)).AND.(T(I+1).LE.MS(CCNRMM))) THEN
	  TAUMS = TAU(I)+
     1            (MS(CCNRMM)-T(I))/(T(I+1)-T(I))*(TAU(I+1)-TAU(I))
	  GOTO 500
	ENDIF
230   CONTINUE
      GOTO 350
C
C       ----------------------------------------------------------------
C       Keine Aufheizung um mehr als 10K
C                                                  >>> W A R N I N G <<<
C310     PRINT*
C	PRINT*,'  keine Aufheizung um mehr als 10K !!!'
310	ERRFLG = 1
	GOTO 400
C       ----------------------------------------------------------------
C       Ac3 nicht erreicht
C                                                  >>> W A R N I N G <<<
C320     PRINT*
C	PRINT*,'  Gefge wird nicht vollst„ndig austenitisiert !!!'
320	ERRFLG = 2
	GOTO 400
C       ----------------------------------------------------------------
C       Kein Schnittpunkt mit ZTA-Diagramm nach 100 Iterationen
C                                                  >>> W A R N I N G <<<
C330     PRINT*
C	PRINT*,'  Schnitt T-Kurve/ZTA: Abbruch nach 100 Iterationen !!!'
330	ERRFLG = 3
	GOTO 400
C       ----------------------------------------------------------------
C       Schmelztemperatur berschritten
C                                                  >>> W A R N I N G <<<
C340     PRINT*
C	PRINT*,'  Schmelztemperatur berschritten !!!'
340	ERRFLG = 4
	GOTO 400
C       ----------------------------------------------------------------
C       Martensitstarttemperatur nicht unterschritten
C                                                      >>> E R R O R <<<
C350     PRINT*
C	PRINT*,'  Martensitstarttemperatur nicht erreicht !!!'
350	ERRFLG = 5
	GOTO 400
C       ----------------------------------------------------------------
C       Starttemperatur zu groá
C                                                      >>> E R R O R <<<
C360     PRINT*
C	PRINT*,'  Erste Temperatur gr”áer (Ac3,unendlich-10K) !!!'
360	ERRFLG = 6
	GOTO 400
C       ----------------------------------------------------------------
C       CCNODE auáerhalb ZTA-Wertebereich
C                                                      >>> E R R O R <<<
C370     PRINT*
C	PRINT*,'  C-Konzentration des Knotens auáerhalb ZTA-Diagramm !!'
370	ERRFLG = 7
	GOTO 400
C       ----------------------------------------------------------------
C       Martensitstarttemp. gr”áer als Ac3,unendlich
C                                                      >>> E R R O R <<<
C380     PRINT*
C	PRINT*,'  Martensitstarttemperatur gr”áer als Ac3,unendlich !!!'
380	ERRFLG = 8
	GOTO 400
C       ----------------------------------------------------------------
C       Temperatur kleiner 1K
C                                                      >>> E R R O R <<<
C       PRINT*
C	PRINT*,'  Temperatur kleiner 1K !!!'
390     ERRFLG = 9
	GOTO 400
C       ----------------------------------------------------------------
C400     CLOSE(10,STATUS='KEEP')
C400	CLOSE(11,STATUS='KEEP')
400	RETURN
C     ==================================================================
C     "T_VERLAUF" schreiben
C
C500   WRITE (11,*,ERR=1100) ' charakteristische Zeiten:'
C      WRITE (11,*,ERR=1100) 'tAc3  ='
C      WRITE (11,'(e15.9)',ERR=1100) TAUAC3
C      WRITE (11,*,ERR=1100) 'tMs   ='
C      WRITE (11,'(e15.9)',ERR=1100) TAUMS
C      WRITE (11,*,ERR=1100) 'tAc3x ='
C      WRITE (11,'(e15.9)',ERR=1100) TAUA3X
C      WRITE (11,*,ERR=1100) ' Temperaturkurve:'
C      WRITE (11,*,ERR=1100) '  t       T'
C      WRITE (11,*,ERR=1100) 'DATA'
C      DO 600 I = 1, NTAU
C	WRITE (11,'(2e15.9)',ERR=1100) TAU(I), T(I)
C600   CONTINUE
C      GOTO 1200
500    RETURN
       END
C
C       ----------------------------------------------------------------
C       Schreibfehler in "T_VERLAUF"
C                                                      >>> E R R O R <<<
C1100    PRINT*
C	PRINT*,'  Fehler beim Schreiben in Datei T_VERLAUF !!!'
C	CLOSE(10,STATUS='KEEP')
C	CLOSE(11,STATUS='KEEP')
C	ERRFLG = 10
C	RETURN
C
C     ------------------------------------------------------------------
C                                                      >>> P R I N T <<<
C1200  CLOSE(10,STATUS='KEEP')
C1200  CLOSE(11,STATUS='KEEP')
C      PRINT*,'beendet.'
C      RETURN
C      END
C
C=======================================================================
