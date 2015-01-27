C                                                                      |
      subroutine dif1dim ( DFILE, TEMPARR, MVN, NODENO, CCNODE, CCMAX,
     1                     LKORN, FMESH, NSTEPS, TEMP1,
     2                     TIME1, TIME2, TAU, EXEMOD, ERRFLG )
C
C     CCNODE = durchschnittliche C-Konzentration des Gefueges
C     CCMAX  = maximale C-Konzentration
C     LKORN  = Korngroesse
C     FMESH  = Anzahl der Unterteilungen in x-Richtung
C     NSTEPS = Anzahl der Zeitschritte
C     TIME1  = Diffusionsstartzeit
C     TIME2  = Diffusionsstopzeit
C     TEMP1  = Temperatur beim Diffusionsstartzeitpunkt
C     DFILE  = Name der Diffusionskoeffizientendatei
C     EXEMOD = Ausfuehrungsmodus
C     ERRFLG = Fehler-Code, = 0 wenn erfolgreich
C              Darf nur vom Hauptprogramm zurckgesetzt werden !!!
C
C ======================================================================
C                        --- Parameter-Datei ---
C     DFILE    = Name der Diffusionskoeffizienten-Datei
C                        --- Diff.koeffizienten-Datei ---
C     NT       = Anzahl Temperaturwerte       (--> x-Werte)
C     NCC      = Anzahl C-Konzentrationswerte (--> y-Werte)
C                (NCC steht nicht explizit in der Datei !)
C     ZEILEN   = Anzahl Zeilen pro Parameterblock (5 Werte pro Zeile)
C     TDIFKO   = Temperaturwerte
C     CCDIFK   = C-Konzentrationswerte
C     D0       = Diffusionskoeffizienten D0=f(TDIFKO,CCDIFK)
C                        --- Hilfsvariablen ---
C     I,J      = Zaehlvariable
C     CLFLG    = Aufruf-Flag
C
      DOUBLE PRECISION CCNODE, CCMAX, LKORN, TEMP1, TDIFKO(100),
     1    CCDIFK(100), D0(100,100), TIME1, TIME2, TAU(500), TVERL(500),
     2    rtimedif, rtimedif1, TEMPARR(5500, 500)
      INTEGER  NT, MVN, NCC, ZEILEN, I, J, EXEMOD, FMESH, NSTEPS, CLFLG,
     1         ERRFLG
      CHARACTER DFILE*120, STRG1*1, STRG2*10, strg3*5
      SAVE TDIFKO, CCDIFK, D0, NT, NCC, CLFLG
C
C     ------------------------------------------------------------------
C     Nur beim ersten Aufruf Werte aus Datei einlesen
C
      IF (CLFLG.EQ.1) THEN
	GOTO 100
      END IF
      CLFLG = 1
C
C     ------------------------------------------------------------------
C     Datei "Diffusionskoeffizient" oeffnen
C
      OPEN (59,FILE=DFILE)
      GOTO 40
C                                                        >>>  E R R O R <<<
30    PAUSE 'Fehler in Parameter-Datei (Name der Diff.koeff.datei) !!!'
      CLOSE(58,STATUS='KEEP')
      RETURN
C     ------------------------------------------------------------------
C     Diffusionskoeffizienten-Datei einlesen:
C       nT einlesen
C
40    READ (59,'(a1)',ERR=90,END=90) STRG1
      IF (STRG1.EQ.'C') THEN
	GOTO 40
      END IF
      BACKSPACE 59
      READ (59,'(20x,i3)',ERR=90,END=90) NT
C                                                     >>>  P R I N T <<<
C      PRINT*,'  Anzahl Temperatur-Werte = ',NT
      ZEILEN=(NT-1)/5+1
C     ------------------------------------------------------------------
C       TDIFKO-Werte einlesen
C
      DO 50 I = 1, ZEILEN
	J = (I-1)*5
	READ ( 59,*,ERR=90,END=90) TDIFKO(J+1),TDIFKO(J+2),TDIFKO(J+3),
     1             TDIFKO(J+4),TDIFKO(J+5)
50    CONTINUE
C     ------------------------------------------------------------------
C       cC- und D-Werte einlesen
C
      NCC=1
      READ (59,'(20x,f10.3)',ERR=90,END=90) CCDIFK(NCC)
C                                                     >>>  P R I N T <<<
C     PRINT*,'Lese D0=f(TDIFKO,cc) fr cc[%] = ',CCDIFK(NCC)
      DO 60 I = 1, ZEILEN
	J = (I-1)*5
	READ (59,*,ERR=90,END=90)
     1  D0((J+1),NCC), D0((J+2),NCC), D0((J+3),NCC),
     2  D0((J+4),NCC),D0((J+5),NCC)
60    CONTINUE
70    READ (59,'(20x,f10.3)',ERR=90,END=85) CCDIFK(NCC+1)
C                                                     >>>  P R I N T <<<
C     PRINT*,'Lese D0=f(TDIFKO,CCDIFK) fr cc[%] = ',CCDIFK(NCC+1)
      NCC = NCC+1
      DO 80 I = 1, ZEILEN
	J = (I-1)*5
	READ (59,*,ERR=90,END=90)
     1  D0((J+1),NCC), D0((J+2),NCC), D0((J+3),NCC),
     2  D0((J+4),NCC),D0((J+5),NCC)
80    CONTINUE
      IF (NCC.LT.100) THEN
	GOTO 70
      END IF
C                                                     >>>  P R I N T <<<
C85    PRINT*,'  Anzahl C-Konzentrationswerte-Werte = ',NCC
C      PRINT*,'Einlesen beendet.'
C      PRINT*
85    CLOSE(59,STATUS='KEEP')
      GOTO 100
C                                                     >>>  E R R O R <<<
90    PAUSE 'Fehler in Diffusionskoeffizienten-Datei !!!'
      CLOSE(59,STATUS='KEEP')
      RETURN
C ======================================================================
c     Datei 'T_VERLAUF' einlesen
c
c
100   i=0
C      rtimedif=mclock()
C      print *,rtime
C      open (10,file="T_VERLAUF")
C110   read (10,'(a5)', err=150, end=150 ) strg3
C      if (strg3.ne.' tAc3') goto 110
C      read (10,'(e15.9)', err=150, end=150 ) TIME1
C                                                     >>>  P R I N T <<<
C      print *,'Diffusionsstartzeit = ',TIME1
C120   read (10,'(a5)', err=150, end=150 ) strg3
C      if (strg3.ne.' tMs ') goto 120
C      read (10,'(e15.9)', err=150, end=150 ) TIME2
C                                                     >>>  P R I N T <<<
C      print *,'Diffusionsstopzeit = ',TIME2
C130   read (10,'(a5)', err=150, end=150 ) strg3
C      if (strg3.ne.' DATA') goto 130
C      i = 0
C140   read (10,*, err=150, end=160 ) TAU(i+1),TVERL(i+1)
C      i=i+1
      DO 140 I=1, MVN
        TVERL(I) = TEMPARR(NODENO, I)
140   CONTINUE
C                                                     >>>  P R I N T <<<
C ======================================================================
c
c     Aufruf von diflin/difkug je nach EXEMOD
c     ------------------------------------------------------------------
      if (EXEMOD.eq.4) then
	 call diflin (CCNODE, CCMAX, LKORN, FMESH, NSTEPS, TIME1, TIME2,
     1      TEMP1, TDIFKO, CCDIFK, D0, TAU, TVERL, NT, MVN, NCC, ERRFLG)
      else if (EXEMOD.eq.5) then
	 call difkug (CCNODE, CCMAX, LKORN, FMESH, NSTEPS, TIME1, TIME2,
     1      TEMP1, TDIFKO, CCDIFK, D0, TAU, TVERL, NT, MVN, NCC, ERRFLG)
      else
	errflg=1
      end if
c      rtimedif1=mclock()
c      print *,rtimedif1
c      rtimedif=rtimedif1-rtimedif
c      print *,'Diffrechenzeit= ',rtimedif
      return
      end
