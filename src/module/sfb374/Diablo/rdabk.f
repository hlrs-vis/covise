C=======================================================================
C     Abkhlkurve einlesen
C     ------------------------------------------------------------------
C     Liest aus "T_VERLAUF" die Abkhlkurve (ab unterschreiten von Ac3x)
C     ein, und speichert sie als Array dT/dt = f(T) im Bereich 
C     1K <= T <= Tschmelz
C     Der Array wird wie folgt belegt:
C     1    <= T <= Tx      :  dT/dt = const. = 0
C     Tx   <= T <= Tmin    :  linear ansteigend von 0 nach dT/dt(Tmin)
C     Tmin <= T <= Ac3x    :  linear interpoliert zw. Sttzstellen
C     Ac3x <= T <= Tschmelz:  dT/dt = const. = dT/dt(Ac3x) 
C     Tx.....Abschrecktemperatur
C     Tmin...niedrigste eingelesene Temperatur
C     Ac3x...Ac3-Temperatur fr t -> unendlich
C     Ist die niedrigste gefundene Temperatur kleiner als Tx, so wird Tx
C     auf Tmin gesetzt
C     ------------------------------------------------------------------
      SUBROUTINE RDABK (TEMP, TAU1, NDID, NT, DT, TSCHM, TX,
     1                  TAUA3X, ERRFLG)
C
C     ------------------------------------------------------------------
C   * AUFRUF-Parameter:
C o   DT       = zu belegender Array dT/dt = f(T)
C i   TSCHM    = Schmelztemperatur = obere Grenze des T-Wertebereichs
C i/o TX       = Abschrecktemperatur
C o   ERRFLG   = Fehler-Code, = 1 wenn Fehler, sonst unver„ndert
C
      DOUBLE PRECISION TEMP(5500,500), TAU1(500), DT(3000),
     1                 TSCHM, TAUA3X, TX
      INTEGER          NT, NODEID, ERRFLG
C
C     ------------------------------------------------------------------
C   * Interne Variablen:
C     TAUA3X   = Beginn der Abkhlkurve (Unterschreiten von Ac3x)
C                aus Datei "T_VERLAUF"
C     TA3X     = Ac3-Temperatur
C     TAUU     = jeweils unterer Zeitsttzpunkt
C     TU       = T(TAUU)
C     TAUO     = jeweils oberer Zeitsttzpunkt
C     TO       = T(TAUO)
C     TMIN     = minimale gefundene Temperatur ab Ac3x, ganzzahlig
C     TMAX     = maximale gefundene Temperatur ab Ac3x, ganzzahlig
C     TT       = Hilfsvariable fr Temperaturwerte
C     TTAU     = Hilfsvariable fr Zeitwerte
C     STRG1,2  = Hilfs-Strings zum Einlesen
C
      DOUBLE PRECISION TA3X, TAUU, TU, TAUO, TO, TTAU
      INTEGER          TT, TMAX, TMIN, I, J
      CHARACTER        STRG1*1, STRG2*5
C
C     ------------------------------------------------------------------
C                                                      >>> P R I N T <<<
C      PRINT*,'Abkhlkurve einlesen'
      TMIN = TSCHM
      TMAX = 1
C
C     ------------------------------------------------------------------
C     Datei "T_VERLAUF" ”ffnen
C
C      OPEN (10,FILE='T_VERLAUF')
C     ------------------------------------------------------------------
C     TAUA3X einlesen
C
C10    READ (10,'(1x,a5)',ERR=20,END=20) STRG1, STRG2
C      IF (STRG1.EQ.'C') THEN
C	GOTO 10
C     END IF
C      IF (STRG2.EQ.'tAc3x') THEN
C	READ (10,*,ERR=20,END=20) TAUA3X
C	GOTO 30
C      END IF
C      GOTO 10
C                                                      >>> E R R O R <<<
C20    PRINT*,'  tAc3x nicht gefunden !!!'
C      CLOSE(10,STATUS='KEEP')
C      ERRFLG = 1
C      RETURN
C     ------------------------------------------------------------------
C     Auf Datenblock spulen
C
C30    READ (10,'(a1,a5)',ERR=40,END=40) STRG1, STRG2
C      IF (STRG1.EQ.'C') THEN
C	GOTO 30
C      END IF
C      IF (STRG2.EQ.'DATA') THEN
C	GOTO 50
C      END IF
C      GOTO 30
C                                                      >>> E R R O R <<<
C40    PRINT*,'  keine T = f(t) Daten gefunden !!!'
C      CLOSE(10,STATUS='KEEP')
C      ERRFLG = 1
C      RETURN
C     ------------------------------------------------------------------
C     dT/dt-Werte jeweils zwischen zwei Sttzstellen ermitteln.
C     Die erste untere Sttzstelle soll TAUA3X,T(TAUA3X) sein 
C     (ggf. interpolieren).
C     ACHTUNG: Der Wertebereich fr T wird NICHT berprft, es werden 
C              auch Temperaturen kleiner 1 oder gr”áer Tschmelz 
C              akzeptiert. Damit keine Probleme entstehen, muá der 
C              T-Wertebereich bereits bei der Erstellung von "T_VERLAUF"
C              geprft werden !!
C
C50    READ (10,*,ERR=100,END=100) TAUO, TO
      I=1
50    TAUO = TAU1(I)
      TO   = TEMP(NDID, I)
      if (TO.LE.1) THEN
        TO = 1
      endif
      IF (TAUO.LE.TAUA3X) THEN
	TAUU = TAUO
	TU = TO
	I=I+1
	GOTO 50
      END IF
      TU = TU + (TAUA3X-TAUU)/(TAUO-TAUU)*(TO-TU)
      TAUU = TAUA3X
      TA3X = TU
C
C     ACHTUNG: TU > TO (Abkhlung !)
C     
      DO 60 J=I+1,NT
        DO 70 TT = TO, TU
          IF (TT .LT. TMIN) THEN
	    TMIN = TT
	  ENDIF
	  IF (TT .GT. TMAX) THEN
	    TMAX = TT
	  ENDIF
	  TTAU = TAUU + (TT-TU)/(TO-TU)*(TAUO-TAUU)
	  if (TT.LE.3000) THEN
	    DT(TT) = (TA3X-TT)/(TTAU-TAUA3X)
	  ENDIF
C                                                      >>> D E B U G <<<
C        IF (DT(TT).LE.0) THEN
C          PRINT*,'negative Abkhlgeschwindigkeit !!!'
C          PRINT*,'dT=,dt=',(TA3X-TT),(TTAU-TAUA3X)
C        ENDIF
70      CONTINUE
      TAUU = TAUO
      TU = TO
C      READ (10,*,ERR=200,END=200) TAUO, TO
      TAUO = TAU1(J)
      TO   = TEMP(NDID, J)
      if (TO.LE.1) THEN
        TO = 1
      ENDIF
60    CONTINUE
C                                                      >>> E R R O R <<<
C100   PRINT*,'  tAc3x im Datenblock nicht gefunden !!!'
C      CLOSE(10,STATUS='KEEP')
C      ERRFLG = 1
C      RETURN
C     ------------------------------------------------------------------
C     Bis Minimum(TX,TMIN) mit "0" auffllen, zwischen TX und TMIN ggf.
C     linear interpolieren.
C
200   IF (TMIN.LE.TX) THEN
	TX = TMIN
      ENDIF
      if (TX.LE.1) THEN
        TX = 1
      ENDIF
      DO 210 TT = 1, TX
        IF (TT.LE.3000) THEN
	  DT(TT) = 0
	ENDIF
210   CONTINUE
      IF (TMIN.GT.TX) THEN
	DO 250 TT = TX, TMIN
	  IF (TT.LE.3000) THEN
    	    DT(TT) = (TT-TX)/(TMIN-TX)*DT(TMIN)
	  ENDIF
250     CONTINUE
      ENDIF
C     ------------------------------------------------------------------
C     Ab TMAX mit dT/dt(TMAX) auffllen
C
      DO 300 TT = TMAX, TSCHM
        IF (TT.LE.3000) THEN
	  DT(TT) = DT(TMAX)
	ENDIF  
300   CONTINUE
C     ------------------------------------------------------------------
C
C                                                      >>> P R I N T <<<
C      PRINT*,'beendet.'
C      CLOSE(10,STATUS='KEEP')
      RETURN
      END
C
C=======================================================================
