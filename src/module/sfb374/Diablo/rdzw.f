C=======================================================================
C     HÑrtewerte fÅr ZwischenstufengefÅge einlesen                                                  
C     ------------------------------------------------------------------
C     Liest das Diagramm Hzw = f((dT/dt),cC) in einen COMMON-Block-Array 
C     (Matrix). Laufvariable sind die normierten, ganzzahlg., Variablen
C     DTNORM: Abschreckgeschwindigkeit
C         1 <= DTNORM <= DTRESZ
C     CCNORM: C-Konzentration 
C         1 <= CCNORM <= CCRESZ
C     Das Diagramm mu· mindestens im Wertebereich
C         0 <=  cC   <= CCMAX
C              dT/dt <= DTMAX
C     vorliegen. DTMAX ist der Maximalwert (dT/dt)kritMax aus der in den
C     Martensitdaten gegebenen Kurve (dT/dt)krit = f(cC). Die HÑrtewerte
C     fÅr C-Konzentrationswerte mit kleinerem (dT/dt)krit mÅssen also
C     entsprechend extrapoliert sein.
C     FÅr (dT/dt)min > 0 wird von (dT/dt)=0 bis (dT/dt)min mit dem
C     jeweiligen HÑrtewert H((dT/dt)min,cC) aufgefÅllt.
C     ------------------------------------------------------------------
      SUBROUTINE RDZW (ZFILE, CCMAX, CCRESZ, DTMAX, DTRESZ, ERRFLG)
C
C     ------------------------------------------------------------------
C   * COMMON-Block /ZWISCH/:
C o   HZWI     = HÑrtewert des ZwischenstufengefÅges
C
      COMMON /ZWISCH/ HZWI(5500,5500)
      INTEGER         HZWI
C
C     ------------------------------------------------------------------
C   * AUFRUF-Parameter:
C i   ZFILE    = Name der ZwischenstufengefÅge-Datei
C i   CCMAX    = hîchster zu berÅcksichtigender C-Konzentrationswert
C i   CCRESZ   = Auflîsung der zu belegenden Matrix bez. C-Konzentration
C i   DTMAX    = max. Abschreckgeschwindigkeit, die verfÅgbar sein mu·
C i   DTRESZ   = Auflsg. der zu belegenden Matrix bez. Abschreckgeschw.
C o   ERRFLG   = Fehler-Code, = 1 wenn Fehler, sonst unverÑndert
C
      CHARACTER        ZFILE*120
      DOUBLE PRECISION CCMAX, DTMAX
      INTEGER          CCRESZ, DTRESZ, ERRFLG
C
C     ------------------------------------------------------------------
C   * interne Variablen:
C     DT       = Array zur Speicherung der eingelesenen dT/dt-Werte
C     CC       = Array zur Speicherung der eingelesenen cC-Werte
C     H        = Array zur Speicherung der eingelesenen HÑrtewerte       
C     NDTGES   = Anzahl dT/dt-Werte = Anzahl x-StÅtzstellen
C     NDT      = Anz. zu berÅcksichtigender dT/dt-Werte (bis (dT/dt)max)
C     NCC      = Anzahl eingelesener cC-Werte = Anzahl y-StÅtzstellen
C     ZEILEN   = Anzahl Zeilen pro Parameterblock (5 Werte pro Zeile)             
C     DTNORM   = normierte Abschreckgeschwindigkeit, ganzzahlig
C     CCNORM   = normierte C-Konzentration, ganzzahlig
C     DT1,DT2  = ZÑhlvariable fÅr normierte Abschreckgeschwindigkeit
C     C1,C2    = ZÑhlvariable fÅr normierte C-Konzentration
C     HHELP    = Hilfsvariable fÅr HÑrtewert
C     I,J      = ZÑhlvariable
C     STRG     = Hilfs-String zum Einlesen
C
      DOUBLE PRECISION DT(100), CC(100), H(100), HHELP
      INTEGER          NDTGES, NDT, NCC, ZEILEN, DTNORM, CCNORM
      INTEGER          DT1, DT2, C1, C2, I, J
      CHARACTER        STRG*1
C
C     ------------------------------------------------------------------
C                                                      >>> P R I N T <<<
C     ------------------------------------------------------------------
C     Datei "ZwischengefÅgedaten" îffnen
C
      OPEN (10,FILE=ZFILE)
C     ------------------------------------------------------------------
C     Auf ersten Datenblock spulen und Anzahl der StÅtzstellen einlesen:
C
10    READ (10,'(a1)',ERR=20,END=20) STRG
      IF (STRG.EQ.'C') THEN
	GOTO 10
      END IF
      BACKSPACE 10
      READ (10,'(20x,i3)',ERR=20,END=20) NDTGES
C                                                      >>> D E B U G <<<
C      PRINT*,'Anzahl Stuetzstellen: ', NDTGES
      IF ((NDTGES.GT.0).AND.(NDTGES.LE.100)) THEN
	ZEILEN=(NDTGES-1)/5+1
	GOTO 30
      END IF
C                                                      >>> E R R O R <<<
20    PRINT*,'  Fehler in der Kopfzeile des dT/dt-Werte-Blocks !!!'
      CLOSE(10,STATUS='KEEP')
      ERRFLG = 1
      RETURN
C     ------------------------------------------------------------------
C     dT/dt-Werte einlesen
C
30    DO 40 I = 1, ZEILEN
	J = (I-1)*5
	READ (10,*,ERR=50,END=50) DT(J+1),DT(J+2),DT(J+3),DT(J+4),
     1                            DT(J+5)
40    CONTINUE
C                                                      >>> D E B U G <<<
C      PRINT*,'Stuetzstellen eingelesen'
      GOTO 60
C                                                      >>> E R R O R <<<
50    PRINT*,'  dT/dt-Werte-Block fehlerhaft/unvollstÑndig !!!'
      CLOSE(10,STATUS='KEEP')
      ERRFLG = 1
      RETURN
C     ------------------------------------------------------------------
C     dT/dt-Wertebereich ÅberprÅfen und
C     Anzahl zu berÅcksichtigender StÅtzstellen ermitteln
C
60    IF(DT(NDTGES).LT.DTMAX) THEN
	GOTO 80
      ENDIF
      DO 70 I = 1, NDTGES
	IF(DT(I).LT.DTMAX) THEN
	  NDT = I
	ENDIF
	IF(DT(I).LT.0) THEN
	  GOTO 80
	ENDIF
70    CONTINUE
      NDT = NDT+1
C                                                      >>> D E B U G <<<
C      PRINT*,'Benutzte Stuetzstellen', NDT
      GOTO 90
C                                                      >>> E R R O R <<<
80    PRINT*,'  dT/dt-Werte-Bereich falsch !!!'
      CLOSE(10,STATUS='KEEP')
      ERRFLG = 1
      RETURN
C     ------------------------------------------------------------------
C     Alle vorhandenen Datenblîcke einlesen (jedoch maximal 100)
C     PrÅfen, ob erster cC-Wert = 0 ist, sonst Fehler.
C
90    READ (10,'(20x,f10.3)',ERR=350,END=350) CC(1)
      IF(CC(1).NE.0) THEN
	GOTO 340
      ENDIF
      NCC = 1
      CCNORM = 1
100   DO 150 I = 1, ZEILEN
	J = (I-1)*5
	READ (10,*,ERR=350,END=350) H(J+1),H(J+2),H(J+3),H(J+4),H(J+5)
150   CONTINUE
      DO 300 J = 1, NDT-1
C                                                      >>> D E B U G <<<
C      PRINT*,'Stuetzstellen H= ',H(J),H(J+1)
	DT1 = 1+DT(J)*(DTRESZ-1)/DTMAX
	DT2 = 1+DT(J+1)*(DTRESZ-1)/DTMAX
	IF (DT1.GE.DT2) THEN
	  GOTO 300
	ENDIF
	DO 200 DTNORM = DT1, DT2
C                                                      >>> D E B U G <<<
C      PRINT*,'Belege Arrays, DTNORM= ',DT1, DT2
	  IF((DTNORM.LE.DTRESZ).AND.(CCNORM.LE.CCRESZ)) THEN
	    HZWI(DTNORM,CCNORM) = 
     1      H(J) + (H(J+1)-H(J))*(DTNORM-DT1)/(DT2-DT1)
C                                                      >>> D E B U G <<<
C      PRINT*,'Hzwisch=',HZWI(DTNORM,CCNORM)
	    IF(HZWI(DTNORM,CCNORM).LT.0) THEN
	      GOTO 360
	    ENDIF
	  ENDIF
200     CONTINUE
300   CONTINUE
      IF((NCC.LT.100).AND.(CC(NCC).LT.CCMAX)) THEN
	READ (10,'(20x,f10.3)',ERR=400,END=400) CC(NCC+1)
	NCC = NCC+1
C                                                      >>> D E B U G <<<
C      PRINT*,'cC = ',CC(NCC)
	CCNORM = 1+CC(NCC)*(CCRESZ-1)/CCMAX
	GOTO 100
      ENDIF
C                                                      >>> D E B U G <<<
C      PRINT*,'Werte pruefen'
      GOTO 400
C                                                      >>> E R R O R <<<
340   PRINT*,'  erster cC-Wert nicht gleich Null !!!'
      CLOSE(10,STATUS='KEEP')
      ERRFLG = 1
      RETURN
C                                                      >>> E R R O R <<<
350   PRINT*,'  ZwischengefÅgedaten fehlerhaft/unvollstÑndig !!!'
      CLOSE(10,STATUS='KEEP')
      ERRFLG = 1
      RETURN
C                                                      >>> E R R O R <<<
360   PRINT*,'  negativer HÑrtewert !!!'
      CLOSE(10,STATUS='KEEP')
      ERRFLG = 1
      RETURN
C     ------------------------------------------------------------------
C     Ist der letzte eingelesene cC-Wert kleiner als CCMAX, dann ist der
C     cC-Wertebereich zu klein.
C
400   IF(CC(NCC).LT.CCMAX) THEN
	GOTO 410
      ENDIF
C                                                      >>> D E B U G <<<
C      PRINT*,'cCMax OK'
      GOTO 500
C                                                      >>> E R R O R <<<
410   PRINT*,'  grî·ter cC-Wert ist kleiner cC,max !!!'
      CLOSE(10,STATUS='KEEP')
      ERRFLG = 1
      RETURN
C     ------------------------------------------------------------------
C     Wenn der letzte eingelesene cC-Wert grî·er als CCMAX ist, dann ist
C     HZWI(...,CCRESZ) noch nicht belegt. Um die ganze Matrix fÅllen zu
C     kînnen, mu· dies aber gewÑhrleistet werden.
C
C     Als untere StÅtzkurve dient die letzte belegte Matrixzeile
C       HZWI(...,CCNORM=?)  
C     Als obere StÅtzkurve dient die zuletzt eingelesene Kurve
C       H(I) = f(DT(I))   1 <= I <= NDT
C
500   IF(CC(NCC).GT.CCMAX) THEN
C                                                      >>> D E B U G <<<
C      PRINT*,'Letzte Zeile belegen'
	CCNORM = 1+CC(NCC-1)*(CCRESZ-1)/CCMAX
	DO 600 I = 1, NDT-1
C                                                      >>> D E B U G <<<
C      PRINT*,'H-Stuetzwert: ',H(I)
	  DT1 = 1+DT(I)*(DTRESZ-1)/DTMAX
	  DT2 = 1+DT(I+1)*(DTRESZ-1)/DTMAX
	  IF (DT1.GE.DT2) THEN
	    GOTO 600
	  ENDIF
	  DO 550 DTNORM = DT1, DT2
	    IF(DTNORM.LE.DTRESZ) THEN
	      HHELP = H(I) + (H(I+1)-H(I))*(DTNORM-DT1)/(DT2-DT1)
C                                                      >>> D E B U G <<<
C      PRINT*,'H-Stuetzwerte: ',HHELP,HZWI(DTNORM,CCNORM)
	      HZWI(DTNORM,CCRESZ) = 
     1        HZWI(DTNORM,CCNORM) + 
     2        (HHELP-HZWI(DTNORM,CCNORM))
     3        *(CCMAX-CC(NCC-1))/(CC(NCC)-CC(NCC-1))
C     2        (CCMAX-CC(NCC-1))/(CC(NCC)-CC(NCC-1))*HHELP
C      PRINT*,'=>: ',HZWI(DTNORM,CCRESZ)
	      IF(HZWI(DTNORM,CCRESZ).LT.0) THEN
		GOTO 360
	      ENDIF
	    ENDIF
550       CONTINUE
600     CONTINUE
      ENDIF
C     ------------------------------------------------------------------
C     Rest der Matrix (also zwischen den cC-StÅtzpunkten) auffÅllen
C     Zwischen dT/dt = 0 und DT(1) mit H(DT(1)) auffÅllen
C
C                                                      >>> D E B U G <<<
C      PRINT*,'Auffuellen'
      DO 730 I = 1, NCC-1
	C1 = 1+CC(I)*(CCRESZ-1)/CCMAX
	C2 = 1+CC(I+1)*(CCRESZ-1)/CCMAX
	IF (C2.GT.CCRESZ) THEN
	  C2 = CCRESZ
	ENDIF
	DO 720 CCNORM = (C1+1), (C2-1)
	  IF(CCNORM.LE.CCRESZ) THEN
	    DO 710 DTNORM = 1, DTRESZ
C                                                      >>> D E B U G <<<
C      PRINT*,'i, Hu,Ho',I,HZWI(DTNORM,C1),HZWI(DTNORM,C2)
C      PRINT*,'c1,c2',C1,C2
	      HHELP = HZWI(DTNORM,C2)-HZWI(DTNORM,C1)
	      HZWI(DTNORM,CCNORM) = 
     1        HZWI(DTNORM,C1) + 
     2        HHELP*(CCNORM-C1)/(C2-C1)
	      IF(HZWI(DTNORM,CCNORM).LT.0) THEN
		GOTO 360
	      ENDIF
710         CONTINUE
	  ENDIF
720     CONTINUE
730   CONTINUE
C     ------------------------------------------------------------------
C     Zwischen dT/dt = 0 und DT(1) mit H(DT(1)) auffÅllen
C
      DTNORM = 1+DT(1)*(DTRESZ-1)/DTMAX
      DO 810 CCNORM = 1, CCRESZ
	DO 800 I = 1, DTNORM
	  HZWI(I,CCNORM) = HZWI(DTNORM,CCNORM)
800     CONTINUE
810   CONTINUE
C     ------------------------------------------------------------------
C
C                                                      >>> P R I N T <<<
      CLOSE(10,STATUS='KEEP')
      RETURN
      END
C
C=======================================================================
