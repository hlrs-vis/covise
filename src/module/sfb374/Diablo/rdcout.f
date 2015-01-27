C=======================================================================
C     C-Konzentrationswerte nach Diffusion einlesen                         
C     ------------------------------------------------------------------
C     Liest aus der FIDAP-Ausgabedatei "diff.FIOUT" alle errechneten
C     C-Konzentrationswerte ein, jedoch maximal NNDMAX Werte.
C     Mit CCMIN und CCMAX k”nnen der erlaubte Minimal- und Maximalwert
C     vorgegeben werden. Werte darunter oder darber werden auf CCMIN
C     bzw. CCMAX gesetzt.
C     ------------------------------------------------------------------
      SUBROUTINE RDCOUT (CC, CCMIN, CCMAX, NND, NNDMAX, ERRFLG)
C
C     ------------------------------------------------------------------
C   * AUFRUF-Parameter:
C o   CC       = zu belegender Array
C i   CCMIN    = erlaubter Minimalwert der C-Konzentration
C i   CCMAX    = erlaubter Maximalwert der C-Konzentration
C o   NND      = Anzahl der tats„chlich eingelesenen Werte
C i   NNDMAX   = Anzahl Werte, die maximal eingelesen werden sollen
C o   ERRFLG   = Fehler-Code, = 1 wenn Fehler, sonst unver„ndert
C
      DOUBLE PRECISION CC(NNDMAX), CCMIN, CCMAX
      INTEGER          NND, NNDMAX, ERRFLG
C
C     ------------------------------------------------------------------
C   * Interne Variablen:
C     MINFLG   = interner Fehlercode, =1: min. C-Konz.wert berschritten
C     MAXFLG   = interner Fehlercode, =1: max. C-Konz.wert berschritten
C     I        = Z„hlvariable
C     STRG     = Hilfs-String zum Einlesen
C
      CHARACTER STRG*10
      INTEGER   MINFLG, MAXFLG, I
C
C     ------------------------------------------------------------------
C                                                      >>> P R I N T <<<
      PRINT*,'C-Konzentrationswerte einlesen'
C     ------------------------------------------------------------------
      MINFLG = 0
      MAXFLG = 0
      NND    = 0
C
C     ------------------------------------------------------------------
C     Datei "diff.FIOUT" ”ffnen
C
      OPEN (10,FILE='diff.FIOUT')
C     ------------------------------------------------------------------
C     'M O V I E' suchen
C     'P R I N T' suchen
C     Knotenanzahl einlesen (=Number of second node, weil die Ergebnisse
C     aller Knoten in diff.FIOUT ausgegeben werden !)
C
10    READ (10,'(1x,a9)',ERR=90,END=90)  STRG
      IF (STRG.NE.'M O V I E') THEN
	GOTO 10
      END IF
20    READ (10,'(1x,a9)',ERR=90,END=90)  STRG
      IF (STRG.NE.'P R I N T') THEN
	GOTO 20
      END IF
30    READ (10,'(71x,a3)',ERR=90,END=90)  STRG
      IF (STRG.NE.'ND2') THEN
	GOTO 30
      END IF
      BACKSPACE 10
      READ (10,'(87x,i4)',ERR=90,END=90) NND
      IF (NND.GT.NNDMAX) THEN
	NND = NNDMAX
	GOTO 100
      ELSEIF (NND.GT.0) THEN
	GOTO 100
      END IF
C                                                      >>> E R R O R <<<
90    PRINT*,'  diff.FIOUT: Fehler im Datenblock-Header !!!'
      CLOSE(10,STATUS='KEEP')
      ERRFLG = 1
      RETURN
C     ------------------------------------------------------------------
C     C-Konzentrationswerte einlesen
C
100   READ (10,'(3x,a4)',ERR=190,END=190) STRG
      IF (STRG.NE.'NODE') THEN
	GOTO 100
      END IF
      DO 110 I = 1, NND
	READ (10,'(50x,e16.10)',ERR=190,END=190) CC(I)
	IF(CC(I).GT.CCMAX) THEN
	  CC(I) = CCMAX
	  MAXFLG  = 1
	END IF
	IF(CC(I).LT.CCMIN) THEN
	  CC(I) = CCMIN
	  MINFLG  = 1
	END IF
110   CONTINUE
C
C     FIDAP gibt nach dem letzten MOVIE-Block keine Movie-Zeit an (!).
C     Das wird ausgenutzt, um den letzten Daten-(MOVIE-)Block zu
C     identifizieren
120   READ (10,'(1x,a10)',ERR=200,END=200) STRG
      IF (STRG.EQ.'MOVIE TIME') THEN
	MAXFLG  = 0
	MINFLG  = 0
	GOTO 100
      END IF
      GOTO 120
C                                                      >>> E R R O R <<<
190   PRINT*,'  diff.FIOUT: Fehler im Datenblock !!!'
      CLOSE(10,STATUS='KEEP')
      ERRFLG = 1
      RETURN
C     ------------------------------------------------------------------
C                                                 >>>  W A R N I N G <<<
200   IF(MINFLG.NE.0) THEN
	PRINT*,'WARNUNG: C-Konzentrationen kleiner cCmin !!!'
      ENDIF
C                                                 >>>  W A R N I N G <<<
      IF(MAXFLG.NE.0) THEN
	PRINT*,'WARNUNG: C-Konzentrationen gr”áer cCmax !!!'
      ENDIF
C     ------------------------------------------------------------------
C
C                                                      >>> P R I N T <<<
      PRINT*,'beendet.'
      CLOSE(10,STATUS='KEEP')
      RETURN
      END
C
C=======================================================================
