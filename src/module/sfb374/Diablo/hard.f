C=======================================================================
C     H„rtewert der Mikrostruktur bestimmen
C     ------------------------------------------------------------------
C     Fr jeden Punkt der Mikrostruktur wird aufgrund der erreichten 
C     C-Konzentr. unter Beachtung der Abkhlkurve und unter Verwendung
C     der Martensit- und Zwischengefgedaten der erreichte H„rtewert 
C     errechnet. Der resultierende H„rtewert der Mikrostruktur wird 
C     als Mittelwert aller H„rtewerte der Mikrostrukturknoten bestimmt
C     ------------------------------------------------------------------
      SUBROUTINE HARD (TEMPARR, ZEIT, NODEID, MVN, H, TSCHM, TX, 
     1                 CCMAX, CCRESM, CCRESZ, DTRESZ, 
     2                 DTMAXZ, EXEMOD, CCDEF, FMESH, TAUA3X, ERRFLG)
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
C   * AUFRUF-Parameter:
C o   H        = zu berechnender H„rtewert der Mikrostruktur
C i   TSCHM    = Schmelztemperatur
C i   TX       = Abschrecktemperatur
C i   CCMAX    = maximal erlaubter C-Konzentrationswert
C i   CCRESM   = Aufl”sung der Martensitdaten-Arrays
C i   CCRESZ   = Aufl”sung des Zwischengefgedaten-Arrays bezgl. cC
C i   DTRESZ   = Aufl”sung des Zwischengefgedaten-Arrays bezgl. dT/dt
C i   DTMAXZ   = max. Abschreckgeschw. des Zwischengefgediagramms
C i   EXEMOD   = Programmausfhrungsmodus, im Modus FAST wurde keine
C                Diffusion gerechnet, zur H„rterechng wird die Default-
C                C-Konzentration CCDEF benutzt
C i   CCDEF    = Default-C-Konzentration
C o   ERRFLG   = Fehler-Code, = 1 wenn Fehler, sonst unver„ndert
C
      DOUBLE PRECISION TEMPARR(5500, 500), ZEIT(500), TSCHM, 
     1                 TX, CCMAX, CCDEF, DTMAXZ, TAUA3X
      INTEGER          H, CCRESM, CCRESZ, DTRESZ, EXEMOD, FMESH,
     1                 NODEID, MVN, ERRFLG
C
C     ------------------------------------------------------------------
C   * Interne Variablen:
C     CC       = cC-Werte der Mikrostruktur, Array
C     NNC      = Anzahl Knoten in der Mikrostruktur = Anzahl cC-Werte
C     DT       = Abkhlkurve dT/dt=f(T), Array
C     CCNRMM   = normierte C-Konzentration fr Martensitdaten
C     CCNRMZ   = normierte C-Konzentration fr Zwischengefgedaten
C     DTNRMZ   = normierte Abkhlgeschwindigkeit fr Zwischengefgedaten
C     HND      = H„rtewert eines Mikrostrukturknotens
C     HSUM     = Summe der H„rtewerte der Mikrostrukturknoten
C                (Typ REAL, da die Summe der H„rtewerte leicht den
C                 Wertebereich von INTEGER-Zahlen (2Byte) bersteigt !)
C     XMART    = Martensitanteil eines Mikrostrukturknotens
C     THELP    = ganzzahlige Hilfstemperatur
C     I        = Z„hlvariable
C
      DOUBLE PRECISION CC(10000), DT(3000), CCMIN
      INTEGER          NNC, CCNRMM, CCNRMZ, DTNRMZ, HND, THELP, I,
     1                 NNCMAX, MRUN, MFAST, MLIN, MKUG
      REAL             HSUM, XMART
C
C                        --- Parameter ---
C     NNCMAX   = max. erlaubte Knotenzahl in der Mikrostruktur = DIM(CC)
C     CCMIN    = minimal erlaubter C-Konzentrationswert == 0.0
C     + Werte fr EXEMOD
C
      PARAMETER (NNCMAX=5500, CCMIN=0.0)
      PARAMETER (MRUN=2, MFAST=3, MLIN=4, MKUG=5)
C
C     ------------------------------------------------------------------
C                                                      >>> P R I N T <<<
C      PRINT*,'H„rtewert der Mikrostruktur berechnen:'
C
C     ------------------------------------------------------------------
C     C-Konzentrationswerte einlesen
C     ==> cC = cC(i)   1 <= i <= NNC
C
      IF (EXEMOD.EQ.MRUN) THEN
	CALL RDCOUT (CC, CCMIN, CCMAX, NNC, NNCMAX, ERRFLG)
C                                                      >>> E R R O R <<<
	IF(ERRFLG.NE.0) THEN
	  RETURN
	ENDIF
      ELSEIF (EXEMOD.EQ.MFAST) THEN
	NNC   = 1
	CC(1) = CCDEF
C        CC(2) = CCDEF
      ELSEIF ((EXEMOD.EQ.MLIN).OR.(EXEMOD.EQ.MKUG)) THEN
	CALL rdc1dd (CC, CCMIN, CCMAX, FMESH, ERRFLG)
	NNC = FMESH+1
C                                                      >>> E R R O R <<<
	IF(ERRFLG.NE.0) THEN
	  RETURN
	ENDIF
      ELSE
	ERRFLG = 1
	RETURN
      ENDIF
C     ------------------------------------------------------------------
C     Abkhlkurve einlesen
C     ==> dT/dt = dT/dt(T)   1 <= T <= TSCHM
C
      CALL RDABK (TEMPARR, ZEIT, NODEID, MVN, DT, TSCHM, TX, 
     1            TAUA3X, ERRFLG)
C                                                      >>> E R R O R <<<
      IF(ERRFLG.NE.0) THEN
	if (ERRFLG.eq.1) then
	  ERRFLG=0
C	  print *,"Es wird die Defaulth„erte von 250HV genommen !"
	  H=250
	end if
	RETURN
      ENDIF
C     ------------------------------------------------------------------
C     Schleife ber alle Mikrostrukturknoten:
C     H„rtewerte HND berechnen, in HSUM aufsummieren
C     (HSUM ist vom Typ REAL, da die Summe von max. 10000 H„rtewerten
C     leicht den Wertebereich von INTEGER-Zahlen (2Byte) bersteigt !)
C
      HSUM = 0.0
      DO 100 I=1, NNC
	CCNRMM = 1+CC(I)*(CCRESM-1)/CCMAX
	THELP  = MS(CCNRMM)
C                                                      >>> D E B U G <<<
C      PRINT*,'Abkhlgeschwindigkeit:',DT(THELP)
C      PRINT*,'krit. Abkhlgeschwindigkeit:',DTKRIT(CCNRMM)
	IF (DT(THELP).GE.DTKRIT(CCNRMM)) THEN
	  IF (TX.LE.MF(CCNRMM)) THEN
	    XMART = 1.0
	  ELSE IF (TX.LE.MS(CCNRMM)) THEN
	    XMART = (MS(CCNRMM)-TX)/(MS(CCNRMM)-MF(CCNRMM))
	  ELSE IF (TX.GT.MS(CCNRMM)) THEN
	    XMART = 0.0
	  ENDIF
C                                                      >>> D E B U G <<<
c      PRINT*,'Martensitfinish:',MF(CCNRMM)
c      PRINT*,'Tunten:',TX
c      PRINT*,'Martensitanteil:',XMART
c      PRINT*,'cC:',CC(I)
c      PRINT*,'cCmax:',CCMAX
c      PRINT*,'cCRes:',CCRESM
c      PRINT*,'cCNrm:',CCNRMM
	  HND = XMART*HMART(CCNRMM)+(1-XMART)*HAUST(CCNRMM)
	ELSE
C                                                      >>> D E B U G <<<
c      PRINT*,'Zwischenstufengefuege'
c      PRINT*,'Abkhlgeschwindigkeit=',DT(THELP)
c      PRINT*,'Martensitstarttemp=',THELP
	  DTNRMZ = 1+DT(THELP)*(DTRESZ-1)/DTMAXZ
	  CCNRMZ = 1+CC(I)*(CCRESZ-1)/CCMAX
	  HND = HZWI(DTNRMZ,CCNRMZ)
	ENDIF
	if (EXEMOD.eq.MKUG) then
	  HSUM = HSUM+HND*(i**3-(i-1)**3)
	else
	  HSUM = HSUM+HND
	end if
100   CONTINUE
      if (EXEMOD.eq.MKUG) then
	 H = HSUM/(NNC**3)
      else
	 H = HSUM/NNC
      end if
C     ------------------------------------------------------------------
C                                                      >>> P R I N T <<<
      RETURN
      END
C
C=======================================================================
