C=======================================================================
C     Materialdaten-Arrays in Excel-Dateien speichern
C     ------------------------------------------------------------------
      SUBROUTINE WRXCEL (CCMAX, DTMAX, CCRESM, CCRESZ, DTRESZ, RESMAX, 
     1                   ERRFLG)
C
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
C   * COMMON-Block /ZWISCH/:
C i   HZWI     = H„rtewert des Zwischenstufengefges
C
      COMMON /ZWISCH/ HZWI(5500,5500)
      INTEGER         HZWI
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
C i   CCMAX    = h”chster C-Konzentrationswert
C i   CCRESM   = Aufl”sung der Martensitdaten-Arrays /MART/
C i   DTMAX    = maximale Abschreckgeschwindigkeit (Hzwi=f(dT/dt,cC))
C i   CCRESZ   = Auflsg. der Zwischengefgematrix /HZWI/ bez. C-Konz.
C i   DTRESZ   = Auflsg. der Zwischengefgematrix /HZWI/ bez. dT/dt
C i   RESMAX   = Anzahl Zeilen der Excel-Dateien, Maximalwert 
C o   ERRFLG   = Fehler-Code, = 1 wenn Fehler, sonst unver„ndert
C
      DOUBLE PRECISION CCMAX, DTMAX
      INTEGER          CCRESM, CCRESZ, DTRESZ, RESMAX, ERRFLG
C
C     ------------------------------------------------------------------
C   * LOKALE Variablen:
C     CC       = C-Konzentration
C     DT       = Aufheiz-/Abschreckgeschwindigkeit
C     TAUHLP   = Hilfs-Array zur Ausgabe ZTA-Diagramm
C     AC3HLP   = Hilfs-Array zur Ausgabe ZTA-Diagramm
C     XCLRES   = Anzahl Zeilen der jeweiligen Excel-Datei
C     IVAR     = variabler Index fr die auszugebenden Arrays
C     JVAR     = variabler Index fr die auszugebenden Arrays
C     I, J     = Laufvariable
C
      DOUBLE PRECISION CC(25), DT, TAUHLP(100), AC3HLP(25)
      INTEGER XCLRES, IVAR(100), JVAR(100), I, J
C
C     ------------------------------------------------------------------
C     Martensitdaten ausgeben in Datei "MART.XLA"
C
      PRINT*,'Schreibe Martensitdaten in MART.XLA'
      XCLRES = RESMAX
      IF (XCLRES.GT.CCRESM) THEN
	XCLRES = CCRESM
      ENDIF
      DO 10 I=1, XCLRES
	IVAR(I) = CCRESM*I/XCLRES
	IF (IVAR(I).LT.1) THEN
	  IVAR(I) = 1
	ENDIF
10    CONTINUE
      OPEN (10,FILE='MART.XLA')
      WRITE (10,100,ERR=200)
      WRITE (10,110,ERR=200) CCMAX
      WRITE (10,120,ERR=200)
      CC(1) = 0.0
      WRITE (10,130,ERR=200)
     1       CC(1), MS(1), MF(1), DTKRIT(1), HMART(1), HAUST(1)
      DO 20 I=1, XCLRES
	CC(1) = IVAR(I)*CCMAX/CCRESM
	WRITE (10,130,ERR=200)
     1  CC(1), MS(IVAR(I)), MF(IVAR(I)), DTKRIT(IVAR(I)),
     2  HMART(IVAR(I)), HAUST(IVAR(I))
20    CONTINUE
      CLOSE(10,STATUS='KEEP')
      GOTO 300
C
C       ----------------------------------------------------------------
C       Schreibformate:
C
100     FORMAT('Martensitdaten:')
110     FORMAT('maximale C-Konzentration [%]: ',e11.3)
120     FORMAT(4x,'cC',9x,'Ms',9x,'Mf',8x,'(dT/dt)k',3x,'Hm',3x,'Ha')
130     FORMAT(4e11.3,1x,2i5)
C       ----------------------------------------------------------------
C       Datei-Schreibfehler
C                                                      >>> E R R O R <<<
200     PRINT*
	PRINT*,'Fehler beim Schreiben in Ausgabedatei MART.XLA'
	CLOSE(10,STATUS='KEEP')
	ERRFLG = 1
	RETURN
C
C     ------------------------------------------------------------------
C     Zwischengefgedaten ausgeben in Datei "ZWIGEF.XLA"
C     Zeilenanzahl ist variabel = XCLRES
C     Spaltenanzahl ist fest = 25 (so far, so FORTRAN)
C
300   PRINT*,'Schreibe Zwischenstufengefgedaten in ZWIGEF.XLA'
      XCLRES = RESMAX
      IF (XCLRES.GT.CCRESZ) THEN
	XCLRES = CCRESZ
      ENDIF
      DO 310 I=1, XCLRES
	IVAR(I) = DTRESZ*I/XCLRES
	IF (IVAR(I).LT.1) THEN
	  IVAR(I) = 1
	ENDIF
310   CONTINUE
      DO 320 J=1, 25
	JVAR(J) = CCRESZ*J/25
	IF (JVAR(J).LT.1) THEN
	  JVAR(J) = 1
	ENDIF
	CC(J) = (JVAR(J)-1)*CCMAX/(CCRESZ-1)
320   CONTINUE
      OPEN (11,FILE='ZWIGEF.XLA')
      WRITE (11,400,ERR=500)
      WRITE (11,410,ERR=500) CCMAX
      WRITE (11,420,ERR=500) DTMAX
      WRITE (11,430,ERR=500)
      WRITE (11,450,ERR=500) (CC(J),J=1,25)
      DO 340 I=1, XCLRES
	DT = (IVAR(I)-1)*DTMAX/(DTRESZ-1)
	WRITE (11,440,ERR=500) DT, (HZWI(IVAR(I),JVAR(J)),J=1,25)
340   CONTINUE
      CLOSE(11,STATUS='KEEP')
      GOTO 600
C
C       ----------------------------------------------------------------
C       Schreibformate:
C
400     FORMAT('Zwischengefgedaten:')
410     FORMAT('maximale C-Konzentration [%]: ',e11.3)
420     FORMAT('maximale Abschreckgeschwindigkeit [K/s]: ',e11.3)
430     FORMAT('Hzwi = f((dT/dt),cC)')
440     FORMAT(e15.3,25i5)
450     FORMAT('  dT/dt  \  cC  ',25e10.3)
C       ----------------------------------------------------------------
C       Datei-Schreibfehler
C                                                      >>> E R R O R <<<
500     PRINT*
	PRINT*,'Fehler beim Schreiben in Ausgabedatei ZWIGEF.XLA'
	CLOSE(11,STATUS='KEEP')
	ERRFLG = 1
	RETURN
C
C     ------------------------------------------------------------------
C     ZTA-Diagramm ausgeben in Datei "ZTA.XLA"
C     Zeilenanzahl ist variabel = XCLRES
C     Spaltenanzahl ist fest = 25 (FORTRAN again !)
C
600   PRINT*,'Schreibe ZTA-Diagramm in ZTA.XLA'
      XCLRES = RESMAX
      DO 610 I=1, XCLRES
	TAUHLP(I) = TAUZTA(1)+(TAUZTA(NTAUZT)-TAUZTA(1))*I/XCLRES
610   CONTINUE
      DO 620 I=1, 25
	CC(I) = CCZTA(1)+(CCZTA(NCCZTA)-CCZTA(1))*I/25
620   CONTINUE
      OPEN (12,FILE='ZTA.XLA')
      WRITE (12,700,ERR=800)
      WRITE (12,730,ERR=800)
      WRITE (12,710,ERR=800) (CC(I),I=1,25)
      DO 630 I=1, 25
	CALL IPOL2P ( TAUZTA, CCZTA, AC3ZTA, TAUZTA(1), CC(I),
     1                AC3HLP(I), NTAUZT, NCCZTA)
630   CONTINUE
      WRITE (12,720,ERR=800) TAUZTA(1), (AC3HLP(I),I=1,25)
      DO 650 I=1, XCLRES
	DO 640 J=1, 25
	  CALL IPOL2P ( TAUZTA, CCZTA, AC3ZTA, TAUHLP(I), CC(J),
     1                  AC3HLP(J), NTAUZT, NCCZTA)
640     CONTINUE
	WRITE (12,720,ERR=800) TAUHLP(I), (AC3HLP(J),J=1,25)
650   CONTINUE
      CLOSE(12,STATUS='KEEP')
      GOTO 10000
C
C       ----------------------------------------------------------------
C       Schreibformate:
C
700     FORMAT('ZTA-Diagramm:')
710     FORMAT('  tau  \  cC    ',25e10.3)
720     FORMAT(e15.3,25f10.3)
730     FORMAT('Ac3 = f(tau,cC)')
C       ----------------------------------------------------------------
C       Datei-Schreibfehler
C                                                      >>> E R R O R <<<
800     PRINT*
	PRINT*,'Fehler beim Schreiben in Ausgabedatei ZTA.XLA'
	CLOSE(12,STATUS='KEEP')
	ERRFLG = 1
	RETURN
C
C     ------------------------------------------------------------------
C
C                                                      >>> P R I N T <<<
10000 PRINT*,'Ausgabe beendet.'
      RETURN
      END
C
C=======================================================================
