C=======================================================================
C     ZTA-Diagramm einlesen                                                  
C     ------------------------------------------------------------------
      SUBROUTINE RDZTA (AFILE, TSCHM, ERRFLG)
C
C     ------------------------------------------------------------------
C   * COMMON-Block /ZTA/:
C i   NTAUZT   = Anzahl Zeit-Sttzstellen des ZTA-Diagramms
C i   NCCZTA   = Anzahl C-Konzentrations-Sttzstellen des ZTA-Diagramms
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
C i   AFILE    = Name der Datei fr Austenitisierungsdiagramm ZTA
C i   TSCHM    = Schmelztemperatur
C o   ERRFLG   = Fehler-Code, = 1 wenn Fehler, sonst unver„ndert
C
      DOUBLE PRECISION TSCHM
      CHARACTER        AFILE*120
      INTEGER          ERRFLG
C
C     ------------------------------------------------------------------
C   * interne Variablen:
C     ZEILEN   = Anzahl Zeilen pro Parameterblock (5 Werte pro Zeile)             
C     I,J      = Z„hlvariable
C     STRG     = Hilfs-String zum Einlesen
C
      INTEGER   ZEILEN, I, J
      CHARACTER STRG*1
C
C     ------------------------------------------------------------------
C                                                      >>> P R I N T <<<
C
C     ------------------------------------------------------------------
C     Datei "ZTA-Diagramm" ”ffnen
C
      OPEN (10,FILE=AFILE)
C
C     ------------------------------------------------------------------
C     ZTA-Diagramm einlesen
C
10    READ (10,'(a1)',ERR=200,END=200) STRG
      IF (STRG.EQ.'C') THEN
        GOTO 10
      END IF
      BACKSPACE 10
      READ (10,'(20x,f10.3)',ERR=200,END=200) TBEZZT
      IF     (TBEZZT.GT.TSCHM) THEN
        GOTO 210
      ELSEIF (TBEZZT.LT.0) THEN
        GOTO 220
      ENDIF
      READ (10,'(20x,i3)',ERR=230,END=230) NTAUZT
      IF ((NTAUZT.LT.1).OR.(NTAUZT.GT.100)) THEN
        GOTO 230
      ENDIF
      ZEILEN=(NTAUZT-1)/5+1
      DO 50 I = 1, ZEILEN
        J = (I-1)*5
        READ (10,*,ERR=240,END=240) 
     1  TAUZTA(J+1),TAUZTA(J+2),TAUZTA(J+3),TAUZTA(J+4),TAUZTA(J+5)
50    CONTINUE
      NCCZTA=1           
      READ (10,'(20x,f10.3)',ERR=240,END=240) CCZTA(NCCZTA)
      DO 60 I = 1, ZEILEN
        J = (I-1)*5
        READ (10,*,ERR=240,END=240) 
     1  AC3ZTA((J+1),NCCZTA), AC3ZTA((J+2),NCCZTA), 
     2  AC3ZTA((J+3),NCCZTA), AC3ZTA((J+4),NCCZTA),
     3  AC3ZTA((J+5),NCCZTA)
60    CONTINUE
70    READ (10,'(20x,f10.3)',ERR=240,END=300) CCZTA(NCCZTA+1)
      NCCZTA = NCCZTA+1
      DO 80 I = 1, ZEILEN
        J = (I-1)*5
        READ (10,*,ERR=240,END=240)
     1  AC3ZTA((J+1),NCCZTA), AC3ZTA((J+2),NCCZTA), 
     2  AC3ZTA((J+3),NCCZTA), AC3ZTA((J+4),NCCZTA),
     3  AC3ZTA((J+5),NCCZTA)
80    CONTINUE
      IF (NCCZTA.LT.100) THEN
        GOTO 70
      END IF
      GOTO 300
C
C       ----------------------------------------------------------------
C       TBEZZT nicht gefunden
C                                                      >>> E R R O R <<<
200     PRINT*,'  Fehler beim Lesen der ZTA-Bezugstemperatur !!!'
        GOTO 290
C       ----------------------------------------------------------------
C       TBEZZT zu groá
C                                                      >>> E R R O R <<<
210     PRINT*,'  ZTA-Bezugstemperatur gr”áer Tschmelz !!!'
        GOTO 290
C       ----------------------------------------------------------------
C       TBEZZT zu klein
C                                                      >>> E R R O R <<<
220     PRINT*,'  ZTA-Bezugstemperatur kleiner 0 !!!'
        GOTO 290
C       ----------------------------------------------------------------
C       NTAUZT falsch
C                                                      >>> E R R O R <<<
230     PRINT*,'  Falsche Sttzstellenanzahl vorgegeben (1,...,100) !!!'
        GOTO 290
C       ----------------------------------------------------------------
C       Lesefehler
C                                                      >>> E R R O R <<<
240     PRINT*,'  Datei-Lesefehler !!!'
        GOTO 290
C       ----------------------------------------------------------------
290     ERRFLG = 1
	CLOSE(10,STATUS='KEEP')
        RETURN
C     ------------------------------------------------------------------
C     Wertebereich berprfen
C
300   DO 310 I = 1, NTAUZT
      IF (TAUZTA(I).LT.0) THEN
        GOTO 400
      ENDIF
310   CONTINUE
      DO 320 I = 1, NCCZTA
      IF (CCZTA(I).LT.0) THEN
        GOTO 410
      ENDIF
320   CONTINUE
      DO 340 I = 1, NTAUZT
        DO 330 J = 1, NCCZTA
          IF     (AC3ZTA(I,J).LT.0) THEN
            GOTO 420
          ELSEIF (AC3ZTA(I,J).GT.TSCHM) THEN
            GOTO 430
          ENDIF
330     CONTINUE
340   CONTINUE
      GOTO 1000
C
C       ----------------------------------------------------------------
C       Zeitwert kleiner Null
C                                                      >>> E R R O R <<<
400     PRINT*,'  Sttzstelle mit Zeitwert kleiner Null !!!'
        GOTO 490
C       ----------------------------------------------------------------
C       C-Konzentrationswert kleiner Null
C                                                      >>> E R R O R <<<
410     PRINT*,'  Sttzstelle mit C-Konzentrationswert kleiner Null !!!'
        GOTO 490
C       ----------------------------------------------------------------
C       Temperaturwert kleiner Null
C                                                      >>> E R R O R <<<
420     PRINT*,'  Sttzstelle mit Temperaturwert kleiner Null !!!'
        GOTO 490
C       ----------------------------------------------------------------
C       Temperaturwert gr”áer Tschmelz
C                                                      >>> E R R O R <<<
430     PRINT*,'  Sttzstelle mit Temperaturwert gr”áer Tschmelz !!!'
        GOTO 490
C       ----------------------------------------------------------------
490     ERRFLG = 1
	CLOSE(10,STATUS='KEEP')
        RETURN
C     ------------------------------------------------------------------
C
C                                                      >>> P R I N T <<<
1000  CLOSE(10,STATUS='KEEP')
      RETURN
      END
C
C=======================================================================
