C=======================================================================
C     Martensitdaten einlesen                                                  
C     ------------------------------------------------------------------
C     Liest die Kurven Ms, Mf, (dT/dt)krit, Hm, Ha = f(cC) in COMMON-
C     Block-Arrays. Laufvariable dieser Arrays ist die normierte, ganz-
C     zahlige, C-Konzentration CCNORM 
C         1 <= CCNORM <= CCRESM
C     Die Kurven mssen mindestens im Wertebereich
C         0 <= cC <= CCMAX
C     vorliegen. Aus (dT/dt)krit wird die maximale kritische Abschreck-
C     geschwindigkeit bestimmt.
C     ------------------------------------------------------------------
      SUBROUTINE RDMART (MFILE, CCMAX, CCRESM, TSCHM, DTMAX, ERRFLG)
C
C     ------------------------------------------------------------------
C   * COMMON-Block /MART/:
C o   MS       = Martensitstarttemperatur
C o   MF       = Martensitfinishtemperatur
C o   DTKRIT   = kritische Abschreckgeschwindigkeit
C o   HMART    = H„rtewert von 100%-igem Martensit
C o   HAUST    = Austenith„rte
C
      COMMON /MART/ MS(5500),MF(5500),DTKRIT(5500),HMART(5500),
     1              HAUST(5500)
      REAL          MS, MF, DTKRIT
      INTEGER       HMART, HAUST
C
C     ------------------------------------------------------------------
C   * AUFRUF-Parameter:
C i   MFILE    = Name der Martensitdaten-Datei
C i   CCMAX    = h”chster zu bercksichtigender C-Konzentrationswert
C i   CCRESM   = Aufl”sung der zu belegenden Arrays
C i   TSCHM    = Schmelztemperatur, maximal zul„ssiger Temperaturwert
C o   DTMAX    = max. Abschreckgeschwindigkeit (zu ermitteln); bis DTMAX
C                mssen sper die Zwischengefgehrten verfgbar sein.
C o   ERRFLG   = Fehler-Code, = 1 wenn Fehler, sonst unver„ndert
C
      CHARACTER        MFILE*120
      DOUBLE PRECISION CCMAX, TSCHM, DTMAX
      INTEGER          CCRESM, ERRFLG
C
C     ------------------------------------------------------------------
C   * interne Variablen:
C     CC       = Array zur Speicherung der eingelesenen cC-Werte       
C     Y        = Array zur Speicherung der eingelesenen Funktionswerte       
C     NCCGES   = Anzahl C-Konzentrationswerte = Anzahl Sttzstellen
C     NCC      = Anzahl zu bercksichtigender Sttzstellen (bis CCMAX)
C     TABNR    = Tabellen-Identifizierer (1,...,5)
C     ZEILEN   = Anzahl Zeilen pro Parameterblock (5 Werte pro Zeile)             
C     CCNORM   = normierte C-Konzentration, ganzzahlig
C     C1,C2    = Z„hlvariable fr normierte C-Konzentration
C     I,J,K    = Z„hlvariable
C     STRG     = Hilfs-String zum Einlesen
C     INTERR   = interner Fehler-Code
C
      DOUBLE PRECISION CC(100), Y(100)
      INTEGER          NCCGES, NCC, TABNR, ZEILEN, CCNORM, C1, C2, 
     1                 I, J, K, INTERR, MSTAB, MFTAB, DTTAB, HMTAB,
     2                 HATAB
      CHARACTER        STRG*1
C
C     ------------------------------------------------------------------
C   * PARAMETER:
C     Kennzeichnung der einzelnen Kurven (Werte fr TABNR)
C     
      PARAMETER (MSTAB=1,MFTAB=2,DTTAB=3,HMTAB=4,HATAB=5)
C
C     ------------------------------------------------------------------
C                                                      >>> P R I N T <<<
      DTMAX  = 0
C
C     ------------------------------------------------------------------
C     Datei "Martensitdaten" ”ffnen
C
      OPEN (10,FILE=MFILE)
C     ------------------------------------------------------------------
C     Auf ersten Datenblock spulen und Anzahl der Sttzstellen einlesen:
C
10    READ (10,'(a1)',ERR=20,END=20) STRG
      IF (STRG.EQ.'C') THEN
	GOTO 10
      END IF
      BACKSPACE 10
      READ (10,'(20x,i3)',ERR=20,END=20) NCCGES
      IF ((NCCGES.GT.0).AND.(NCCGES.LE.100)) THEN
	ZEILEN=(NCCGES-1)/5+1
	GOTO 30
      END IF
C                                                      >>> E R R O R <<<
20    PRINT*,'  Fehler in der Kopfzeile des cC-Werte-Blocks !!!'
      CLOSE(10,STATUS='KEEP')
      ERRFLG = 1
      RETURN
C     ------------------------------------------------------------------
C     cC-Werte einlesen
C
30    DO 40 I = 1, ZEILEN
	J = (I-1)*5
	READ (10,*,ERR=50,END=50) CC(J+1),CC(J+2),CC(J+3),CC(J+4),
     1                            CC(J+5)
40    CONTINUE
      GOTO 60
C                                                      >>> E R R O R <<<
50    PRINT*,'  cC-Werte-Block fehlerhaft/unvollst„ndig !!!'
      CLOSE(10,STATUS='KEEP')
      ERRFLG = 1
      RETURN
C     ------------------------------------------------------------------
C     cC-Wertebereich berprfen und
C     Anzahl zu bercksichtigender Sttzstellen ermitteln
C
60    IF(CC(1).NE.0) THEN
	GOTO 80
      ENDIF
      IF(CC(NCCGES).LT.CCMAX) THEN
	GOTO 80
      ENDIF
      DO 70 I = 1, NCCGES
	IF(CC(I).LT.CCMAX) THEN
	  NCC = I
	ENDIF
	IF(CC(I).LT.0) THEN
	  GOTO 80
	ENDIF
70    CONTINUE
      NCC = NCC+1
      GOTO 90
C                                                      >>> E R R O R <<<
80    PRINT*,'  cC-Werte-Bereich falsch !!!'
      CLOSE(10,STATUS='KEEP')
      ERRFLG = 1
      RETURN
C     ------------------------------------------------------------------
C     5 Kurven einlesen; Kurven anhand TABNR identifizieren und auf den
C     entsprechenden COMMON-Array schreiben
C     kleiner Trick:
C       um festzustellen, ob jede Kurve einmal eingelesen wurde, wird
C       INTERR auf 1*1+2*2+3*3+4*4+5*5 = 55 gesetzt und dann bei jeder
C       Kurve um TABNR**2 reduziert. Ergebnis muá INTERR=0 sein
C
90    INTERR = 55
      DO 300 I = 1, 5
	READ (10,'(20x,i10)',ERR=350,END=350) TABNR
C                                                      >>> D E B U G <<<
C      PRINT*,'Tabelle: ',TABNR
	DO 100 J = 1, ZEILEN
	  K = (J-1)*5
	  READ (10,*,ERR=350,END=350) Y(K+1),Y(K+2),Y(K+3),Y(K+4),Y(K+5)
100     CONTINUE
	INTERR = INTERR-TABNR**2
	DO 200 J = 1, NCC-1
	  C1 = 1+CC(J)*(CCRESM-1)/CCMAX
	  C2 = 1+CC(J+1)*(CCRESM-1)/CCMAX
	  IF (C1.GE.C2) THEN
	    GOTO 200
	  ENDIF
	  DO 150 CCNORM = C1, C2
	    IF(CCNORM.LE.CCRESM) THEN
C                                                      >>> D E B U G <<<
C      PRINT*,'Belege Arrays, CCNORM= ',C1, C2
	      IF(TABNR.EQ.MSTAB) THEN
		MS(CCNORM) = Y(J) +
     1                       (Y(J+1)-Y(J))*(CCNORM-C1)/(C2-C1)
		IF((MS(CCNORM).LT.1).OR.(MS(CCNORM).GT.TSCHM)) THEN
		  GOTO 360
		ENDIF
	      ENDIF
	      IF(TABNR.EQ.MFTAB) THEN
		MF(CCNORM) = Y(J) +
     1                       (Y(J+1)-Y(J))*(CCNORM-C1)/(C2-C1)
		IF((MF(CCNORM).LT.1).OR.(MF(CCNORM).GT.TSCHM)) THEN
		  GOTO 370
		ENDIF
	      ENDIF
	      IF(TABNR.EQ.DTTAB) THEN
		DTKRIT(CCNORM) = Y(J) + 
     1                          (Y(J+1)-Y(J))*(CCNORM-C1)/(C2-C1)
		IF(DTKRIT(CCNORM).GT.DTMAX) THEN
		  DTMAX = DTKRIT(CCNORM)
		ENDIF
		IF(DTKRIT(CCNORM).LT.0) THEN
		  GOTO 380
		ENDIF
	      ENDIF
	      IF(TABNR.EQ.HMTAB) THEN
		HMART(CCNORM) = Y(J) +
     1                          (Y(J+1)-Y(J))*(CCNORM-C1)/(C2-C1)
C                                                      >>> D E B U G <<<
C      PRINT*,'i,Hmart=',CCNORM,HMART(CCNORM)
		IF(HMART(CCNORM).LT.0) THEN
		  GOTO 390
		ENDIF
	      ENDIF
	      IF(TABNR.EQ.HATAB) THEN
		HAUST(CCNORM) = Y(J) + 
     1                          (Y(J+1)-Y(J))*(CCNORM-C1)/(C2-C1)
		IF(HAUST(CCNORM).LT.0) THEN
		  GOTO 390
		ENDIF
	      ENDIF
	    ENDIF
150       CONTINUE
200     CONTINUE
300   CONTINUE
      IF(INTERR.EQ.0) THEN
	GOTO 500
      ENDIF
C                                                      >>> E R R O R <<<
350   PRINT*,'  Martensitdaten fehlerhaft/unvollst„ndig !!!'
      CLOSE(10,STATUS='KEEP')
      ERRFLG = 1
      RETURN
C                                                      >>> E R R O R <<<
360   PRINT*,'  Ms kleiner 1K oder gr”áer Tschmelz !!!'
      CLOSE(10,STATUS='KEEP')
      ERRFLG = 1
      RETURN
C                                                      >>> E R R O R <<<
370   PRINT*,'  Mf kleiner 1K oder gr”áer Tschmelz !!!'
      CLOSE(10,STATUS='KEEP')
      ERRFLG = 1
      RETURN
C                                                      >>> E R R O R <<<
380   PRINT*,'  negative kritische Abkhlgeschwindigkeit !!!'
      CLOSE(10,STATUS='KEEP')
      ERRFLG = 1
      RETURN
C                                                      >>> E R R O R <<<
390   PRINT*,'  negativer H„rtewert !!!'
      CLOSE(10,STATUS='KEEP')
      ERRFLG = 1
      RETURN
C     ------------------------------------------------------------------
C
C                                                      >>> P R I N T <<<
500   CLOSE(10,STATUS='KEEP')
      RETURN
      END
C
C=======================================================================
