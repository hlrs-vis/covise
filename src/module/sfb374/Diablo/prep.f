C                                                                      |
C=======================================================================
C     FIDAP-Ausgabedatei "temp.FIOUT" aufbereiten
C     ------------------------------------------------------------------
C     Liest aus der FIDAP-Ausgabedatei "???", fr welche 
C     MAKROSKOPISCHEN
C     Knoten die H„rteberechnung erfolgen soll. 
C     Der Array NODEID(i) wird mit den Knoten-"ID"s (= FIDAP-Knoten-
C     nummern) dieser Knoten belegt, die Arrays XND(i), YND(i), ZND(i)
C     mit den Koordinaten der Knoten.
C     Die Datenzeilen dieser Knoten werden aus "temp.FIOUT" nach 
C     "T_VERLAEUFE" kopiert.
C     Die zugeh”rigen Zeiten werden im Array TAU[j] gespeichert.
C     Die Anzahl zu bearbeitender Knoten NNODES sowie die Anzahl 
C     vorhandener Zeitsttzstellen NTAU wird bestimmt.
C     ------------------------------------------------------------------
      SUBROUTINE PREP (DPATH1, DPATH2, NODEID, XND, YND, ZND, NNODES, 
     1                 NNDMAX, TAU, NTAU, NTMAX, TEMPARR, ERRFLG)
C
C     ------------------------------------------------------------------
C   * AUFRUF-Parameter:
C o   NODEID   = Array mit FIDAP-Knotennummern der relevanten Knoten
C o   XND      = x-Koordinaten der zu betrachtenden Knoten
C o   YND      = y-Koordinaten der zu betrachtenden Knoten
C o   ZND      = z-Koordinaten der zu betrachtenden Knoten
C o   NNODES   = Anzahl relevanter Knoten
C i   NNDMAX   = maximal erlaubte Anzahl relevanter Knoten
C o   TAU      = Array mit eingelesenen Zeiten (MOVIE-Zeiten)
C o   NTAU     = Anzahl eingelesener Zeiten
C i   NTMAX    = maximal erlaubte Anzahl Zeiten
C o   TEMPARR  = Speichert die TEmperaturkurven pro Knoten
C                vermeidet Festplattenzugriffe auf T_VERLAUFE
C                --> dauert lang 
C o   ERRFLG   = Fehler-Code, = 1 wenn Fehler, sonst unver„ndert
C
      CHARACTER DPATH1*120, DPATH2*120
      DOUBLE PRECISION TAU(500), XND(5500), YND(5500), ZND(5500),
     1                 TEMPARR(5500, 500)
C      INTEGER          NTMAX, NNDMAX
C     1                 NODEID(NNDMAX), NNODES, NTAU, ERRFLG
      INTEGER          NTMAX, NNDMAX,
     1                 NODEID(5500), NNODES, NTAU, ERRFLG
C      
C
C     -----------------------------------------------------------------
C   * Interne Variablen:
C     X,Y,Z    = Koordinaten des aktuellen Knotens
C     T        = Temperaturwert des aktuellen Knotens
C     NNDBLK   = Anzahl der Knoten der aktuellen Entity (FIOUT.nodes)
C     ID       = FIDAP-Nummer des aktuellen Knotens
C     ND1,ND2  = FIDAP-Nummern des ersten und des letzten Knotens im 
C                "MOVIE-Block"
C     NND      = Anzahl der im "MOVIE-Block" gefundenen relevanten 
C                Knoten (MUSS gleich NNODES sein, sonst ist MOVIE-Block
C                        unvollst„ndig)
C     IDS      = Array zum Zwischenspeichern der FIDAP-Knotennummern
C     I        = Z„hlvariable
C     STRG     = Hilfs-String zum Einlesen
C 

      DOUBLE PRECISION X, Y, Z, T
      CHARACTER STRG*20, STRG2*20
C      INTEGER   ID, ND1, ND2, NND, IDS(NNDMAX), I
      INTEGER   NELBLK, NNDBLK, ID, ND1, ND2, NND, IDS(5500), IDS1(100),
     1       IDS2(100), IDS3(100), IDS4(100), IDS5(100), IDS6(100), 
     2       IDS7(100), IDS8(100), I, J, ND, EQFLG1,  EQFLG2, EQFLG3, 
     3       EQFLG4, EQFLG5, EQFLG6, EQFLG7, EQFLG8

C
C     ------------------------------------------------------------------
C                                                      >>> P R I N T <<<
C      PRINT*,'Rechnung wird vorbereitet'
      NNODES = 0
      NTAU   = 0
C
C     ------------------------------------------------------------------
C     Dateien ”ffnen
C
      OPEN (10,FILE=DPATH1)
      OPEN (11,FILE=DPATH2)
C
C     ==================================================================
C     >>> D U M M Y <<<                                >>> D U M M Y <<<
C
C     Einlesen, welche Knoten betrachtet werden sollen; Fehler, wenn
C     maximale Knotenanzahl NNDMAX berschritten wird.
C      
C     !!!   Ersetzen durch Einlesen der relevanten Knoten in IDS(i)  !!!
C
C     ------------------------------------------------------------------
C     MESH-DATA-Block in "temp.FIOUT.nodes" suchen
C
100   READ (10,'(1x,a18)',ERR=110,END=110) STRG
      IF (STRG.NE.'T O T A L  M E S H') THEN
	GOTO 100
      END IF
      GOTO 120
C                                                      >>> E R R O R <<<
110   PRINT*,'  MESH-DATA-Block in temp.FIOUT.nodes nicht gefunden !!!'
      CLOSE(10,STATUS='KEEP')
      CLOSE(11,STATUS='KEEP')
      ERRFLG = 1
      RETURN
C     ------------------------------------------------------------------
C     MESH-DATA-Info einlesen
C
120   READ (10,'(16x,a15)',ERR=200,END=200) STRG
      IF (STRG.NE.'NO. OF ELEMENTS') THEN
	GOTO 120
      END IF
130   BACKSPACE 10
      READ (10,'(32x,i6,15x,i6)',ERR=200,END=500) NELBLK,NNDBLK
135   READ (10,'(5x,a5,6x,a15)',ERR=200,END=500) STRG, STRG2
      IF (STRG.EQ.'LOCAL') THEN
	GOTO 140
      ELSEIF (STRG2.EQ.'NO. OF ELEMENTS') THEN
	GOTO 130
      ELSE
	GOTO 135
      END IF
140   NNODES = NNODES+NELBLK*NNDBLK
      IF (NNODES.GT.NNDMAX) THEN
        GOTO 210
      END IF
      NNODES = NNODES-NELBLK*NNDBLK
      if (NNDBLK.eq.3) then
	DO 142 I=1,NELBLK
	  READ (10,'(21x,3i8)',ERR=200,END=500) IDS1(I),IDS2(I),IDS3(I)
142     CONTINUE
	do 144,J=1,NELBLK
	  EQFLG1=0
	  EQFLG2=0
	  EQFLG3=0
	  do 143,I=0,NNODES-1
	    if (IDS(I+1).eq.IDS1(J)) then
	      EQFLG1=1
	    end if
	    if (IDS(I+1).eq.IDS2(J)) then
	      EQFLG2=1
	    end if
	    if (IDS(I+1).eq.IDS3(J)) then
	      EQFLG3=1
	    endif
143       continue
	  if (EQFLG1.eq.0) then
	    IDS(NNODES+1)=IDS1(J)
	    NNODES=NNODES+1
	    IF (NNODES.GT.NNDMAX) THEN
	       GOTO 210
	    END IF
	  end if
	  if (EQFLG2.eq.0) then
	    IDS(NNODES+1)=IDS2(J)
	    NNODES=NNODES+1
	    IF (NNODES.GT.NNDMAX) THEN
	       GOTO 210
	    END IF
	  end if
	  if (EQFLG3.eq.0) then
	    IDS(NNODES+1)=IDS3(J)
	    NNODES=NNODES+1
	    IF (NNODES.GT.NNDMAX) THEN
	       GOTO 210
	    END IF
	  endif
144     continue
      elseif (NNDBLK.eq.4) then
	DO 145 I=1,NELBLK
	  READ (10,'(21x,4i8)',ERR=200,END=500) IDS1(I),IDS2(I),IDS3(I),
     1                                          IDS4(I)
145     CONTINUE
	do 147,J=1,NELBLK
	  EQFLG1=0
	  EQFLG2=0
	  EQFLG3=0
	  EQFLG4=0
	  do 146,I=0,NNODES-1
	    if (IDS(I+1).eq.IDS1(J)) then
	      EQFLG1=1
	    end if
	    if (IDS(I+1).eq.IDS2(J)) then
	      EQFLG2=1
	    end if
	    if (IDS(I+1).eq.IDS3(J)) then
	      EQFLG3=1
	    end if
	    if (IDS(I+1).eq.IDS4(J)) then
	      EQFLG4=1
	    endif
146       continue
	  if (EQFLG1.eq.0) then
	    IDS(NNODES+1)=IDS1(J)
	    NNODES=NNODES+1
	    IF (NNODES.GT.NNDMAX) THEN
	       GOTO 210
	    END IF
	  end if
	  if (EQFLG2.eq.0) then
	    IDS(NNODES+1)=IDS2(J)
	    NNODES=NNODES+1
	    IF (NNODES.GT.NNDMAX) THEN
	       GOTO 210
	    END IF
	  end if
	  if (EQFLG3.eq.0) then
	    IDS(NNODES+1)=IDS3(J)
	    NNODES=NNODES+1
	    IF (NNODES.GT.NNDMAX) THEN
	       GOTO 210
	    END IF
	  end if
	  if (EQFLG4.eq.0) then
	    IDS(NNODES+1)=IDS4(J)
	    NNODES=NNODES+1
	    IF (NNODES.GT.NNDMAX) THEN
	       GOTO 210
	    END IF
	  endif
147     continue
      elseif (NNDBLK.eq.6) then
	DO 148 I=1,NELBLK
	  READ (10,'(21x,6i8)',ERR=200,END=500) IDS1(I),IDS2(I),IDS3(I),
     1                                          IDS4(I),IDS5(I),IDS6(I)
148     CONTINUE
	do 150,J=1,NELBLK
	  EQFLG1=0
	  EQFLG2=0
	  EQFLG3=0
	  EQFLG4=0
	  EQFLG5=0
	  EQFLG6=0
	  do 149,I=0,NNODES-1
	    if (IDS(I+1).eq.IDS1(J)) then
	      EQFLG1=1
	    end if
	    if (IDS(I+1).eq.IDS2(J)) then
	      EQFLG2=1
	    end if
	    if (IDS(I+1).eq.IDS3(J)) then
	      EQFLG3=1
	    end if
	    if (IDS(I+1).eq.IDS4(J)) then
	      EQFLG4=1
	    end if
	    if (IDS(I+1).eq.IDS5(J)) then
	      EQFLG5=1
	    end if
	    if (IDS(I+1).eq.IDS6(J)) then
	      EQFLG6=1
	    endif
149       continue
	  if (EQFLG1.eq.0) then
	    IDS(NNODES+1)=IDS1(J)
	    NNODES=NNODES+1
	    IF (NNODES.GT.NNDMAX) THEN
	       GOTO 210
	    END IF
	  end if
	  if (EQFLG2.eq.0) then
	    IDS(NNODES+1)=IDS2(J)
	    NNODES=NNODES+1
	    IF (NNODES.GT.NNDMAX) THEN
	       GOTO 210
	    END IF
	  end if
	  if (EQFLG3.eq.0) then
	    IDS(NNODES+1)=IDS3(J)
	    NNODES=NNODES+1
	    IF (NNODES.GT.NNDMAX) THEN
	       GOTO 210
	    END IF
	  end if
	  if (EQFLG4.eq.0) then
	    IDS(NNODES+1)=IDS4(J)
	    NNODES=NNODES+1
	    IF (NNODES.GT.NNDMAX) THEN
	       GOTO 210
	    END IF
	  end if
	  if (EQFLG5.eq.0) then
	    IDS(NNODES+1)=IDS5(J)
	    NNODES=NNODES+1
	    IF (NNODES.GT.NNDMAX) THEN
	       GOTO 210
	    END IF
	  end if
	  if (EQFLG6.eq.0) then
	    IDS(NNODES+1)=IDS6(J)
	    NNODES=NNODES+1
	    IF (NNODES.GT.NNDMAX) THEN
	       GOTO 210
	    END IF
	  endif
150     continue
      elseif (NNDBLK.eq.8) then
	DO 151 I=1,NELBLK
	  READ (10,'(21x,8i8)',ERR=200,END=500) IDS1(I),IDS2(I),IDS3(I),
     1                          IDS4(I),IDS5(I),IDS6(I),IDS7(I),IDS8(I)
151     CONTINUE
	do 153,J=1,NELBLK
	  EQFLG1=0
	  EQFLG2=0
	  EQFLG3=0
	  EQFLG4=0
	  EQFLG5=0
	  EQFLG6=0
	  EQFLG7=0
	  EQFLG8=0
	  do 152,I=0,NNODES-1
	    if (IDS(I+1).eq.IDS1(J)) then
	      EQFLG1=1
	    end if
	    if (IDS(I+1).eq.IDS2(J)) then
	      EQFLG2=1
	    end if
	    if (IDS(I+1).eq.IDS3(J)) then
	      EQFLG3=1
	    end if
	    if (IDS(I+1).eq.IDS4(J)) then
	      EQFLG4=1
	    end if
	    if (IDS(I+1).eq.IDS5(J)) then
	      EQFLG5=1
	    end if
	    if (IDS(I+1).eq.IDS6(J)) then
	      EQFLG6=1
	    end if
	    if (IDS(I+1).eq.IDS7(J)) then
	      EQFLG7=1
	    end if
	    if (IDS(I+1).eq.IDS8(J)) then
	      EQFLG8=1
	    endif
152       continue
c          pause
	  if (EQFLG1.eq.0) then
	    IDS(NNODES+1)=IDS1(J)
	    NNODES=NNODES+1
	    IF (NNODES.GT.NNDMAX) THEN
	       GOTO 210
	    END IF
	  end if
	  if (EQFLG2.eq.0) then
	    IDS(NNODES+1)=IDS2(J)
	    NNODES=NNODES+1
	    IF (NNODES.GT.NNDMAX) THEN
	       GOTO 210
	    END IF
	  end if
	  if (EQFLG3.eq.0) then
	    IDS(NNODES+1)=IDS3(J)
	    NNODES=NNODES+1
	    IF (NNODES.GT.NNDMAX) THEN
	       GOTO 210
	    END IF
	  end if
	  if (EQFLG4.eq.0) then
	    IDS(NNODES+1)=IDS4(J)
	    NNODES=NNODES+1
	    IF (NNODES.GT.NNDMAX) THEN
	       GOTO 210
	    END IF
	  end if
	  if (EQFLG5.eq.0) then
	    IDS(NNODES+1)=IDS5(J)
	    NNODES=NNODES+1
	    IF (NNODES.GT.NNDMAX) THEN
	       GOTO 210
	    END IF
	  end if
	  if (EQFLG6.eq.0) then
	    IDS(NNODES+1)=IDS6(J)
	    NNODES=NNODES+1
	    IF (NNODES.GT.NNDMAX) THEN
	       GOTO 210
	    END IF
	  end if
	  if (EQFLG7.eq.0) then
	    IDS(NNODES+1)=IDS7(J)
	    NNODES=NNODES+1
	    IF (NNODES.GT.NNDMAX) THEN
	       GOTO 210
	    END IF
	  end if
	  if (EQFLG8.eq.0) then
	    IDS(NNODES+1)=IDS8(J)
	    NNODES=NNODES+1
	    IF (NNODES.GT.NNDMAX) THEN
	       GOTO 210
	    END IF
	  endif
153     continue
      end if
      goto 135
c165   READ (10,'(16x,a15)',ERR=500,END=500) STRG
c      IF (STRG.NE.'NO. OF ELEMENTS') THEN
c        GOTO 165
c      END IF
c170   BACKSPACE 10
c      READ (10,'(32x,i6)',ERR=500,END=500) NNDBLK
c175   READ (10,'(5x,a5,6x,a15)',ERR=500,END=500) STRG, STRG2
c      IF (STRG.EQ.'LOCAL') THEN
c        GOTO 180
c      ELSEIF (STRG2.EQ.'NO. OF ELEMENTS') THEN
c        GOTO 170
c      ELSE
c        GOTO 175
c      END IF
c180   NNODES = NNODES + NNDBLK
c      IF (NNODES.GT.NNDMAX) THEN
c        GOTO 210
c      END IF
c      DO 190 I=(NNODES-NNDBLK+1),NNODES
c        READ (10,'(23x,i6)',ERR=200,END=200) IDS(I)
c190   CONTINUE
c      GOTO 175
C                                                      >>> E R R O R <<<
200   PRINT*,'  MESH-DATA-Block in temp.FIOUT.nodes unvollst„ndig !!!'
      CLOSE(10,STATUS='KEEP')
      CLOSE(11,STATUS='KEEP')
      ERRFLG = 1
      RETURN
C                                                      >>> E R R O R <<<
210   PRINT*,'  Berechnung fr mehr als 5500 Knoten nicht m”glich !!!'
      print *,nnodes
      CLOSE(10,STATUS='KEEP')
      CLOSE(11,STATUS='KEEP')
      ERRFLG = 1
      RETURN
C
C     >>> D U M M Y <<<                                >>> D U M M Y <<<
C     ==================================================================
C
C     ------------------------------------------------------------------
C     MOVIE-Block in "temp.FIOUT" suchen
C
500   READ (11,'(1x,a9)',ERR=510,END=510) STRG
      IF (STRG.NE.'M O V I E') THEN
	GOTO 500
      END IF
C      print *,nnodes
      GOTO 520
C                                                      >>> E R R O R <<<
510   PRINT*,'  MOVIE-Block in temp.FIOUT nicht gefunden !!!'
C      print *,nnodes
      CLOSE(10,STATUS='KEEP')
      CLOSE(11,STATUS='KEEP')
      ERRFLG = 1
      RETURN
C     ------------------------------------------------------------------
C     MOVIE-Block-Header einlesen
C
520   READ (11,'(71x,a3)',ERR=540,END=540) STRG
      IF (STRG.NE.'ND1') THEN
	GOTO 520
      END IF
      BACKSPACE 11
      READ (11,'(87x,i4)',ERR=540,END=540) ND1
530   READ (11,'(71x,a3)',ERR=540,END=540) STRG
      IF (STRG.NE.'ND2') THEN
	GOTO 530
      END IF
      BACKSPACE 11
      READ (11,'(87x,i4)',ERR=540,END=540) ND2
C      GOTO 550
      GOTO 600
C                                                      >>> E R R O R <<<
540   PRINT*,'  MOVIE-Block-Header in temp.FIOUT unvollst„ndig !!!'
      CLOSE(10,STATUS='KEEP')
      CLOSE(11,STATUS='KEEP')
      ERRFLG = 1
      RETURN
C     ------------------------------------------------------------------
C     Alle Zeitbl”cke des MOVIE-Datenblocks einlesen, jedoch maximal
C     NTMAX. Datenzeilen der zu betrachtenden Knoten in "T_VERLAEUFE"
C     kopieren. Gleichzeitig wird NODEID(i) aufgefllt, und zwar in der
C     Reihenfolge, wie die ausgesuchten Knoten im MOVIE-Block vorkommen.
C
C550   OPEN (12,FILE='T_VERLAEUFE')
C
600   READ (11,'(3x,a4)',ERR=1000,END=1000) STRG
      IF (STRG.NE.'NODE') THEN
	GOTO 600
      END IF
      NND = 0
      DO 650 ND = ND1, ND2
	READ (11,'(3x,i4,3e14.6,e17.8)',ERR=910,END=910)
     1       ID, X, Y, Z, T
	DO 610 I = 1, NNODES
	  IF (ID.EQ.IDS(I)) THEN
	    NND = NND+1
	    NODEID(NND) = ID
	    XND(NND)    = X
	    YND(NND)    = Y
	    ZND(NND)    = Z
C	    WRITE (12,'(3x,i4,3e14.6,e17.8)',ERR=900)
C     1             ID, X, Y, Z, T
            TEMPARR(NND, NTAU+1) = T
	  ENDIF
610     CONTINUE
650   CONTINUE
      IF (NND.NE.NNODES) THEN
	GOTO 810
      END IF
700   READ (11,'(1x,a10)',ERR=1000,END=1000) STRG
      IF (STRG.NE.'MOVIE TIME') THEN
	GOTO 700
      END IF
      NTAU = NTAU + 1
      IF (NTAU.GT.NTMAX) THEN
	GOTO 800
      ENDIF
      BACKSPACE 11
      READ (11,'(13x,e11.3)',ERR=910,END=910) TAU(NTAU)
      GOTO 600
C
C       ----------------------------------------------------------------
C       NTAU > NTMAX
C                                                      >>> E R R O R <<<
800     PRINT*
	PRINT*,'  Anzahl Zeitschritte in temp.FIOUT zu groá !!!'
	GOTO 990
C       ----------------------------------------------------------------
C       Nicht fr jeden zu betrachtenden Knoten eine Datenzeile
C       gefunden
C                                                      >>> E R R O R <<<
810     PRINT*
	PRINT*,'  temp.FIOUT unvollst„ndig !!!'
	GOTO 990
C       ----------------------------------------------------------------
C       Datei-Schreibfehler
C                                                      >>> E R R O R <<<
900     PRINT*
	PRINT*,'  Fehler beim Schreiben in Datei T_VERLAEUFE !!!'
	GOTO 990
C       ----------------------------------------------------------------
C       Datei-Lesefehler
C                                                      >>> E R R O R <<<
910     PRINT*
	PRINT*,'  Lesefehler innerhalb temp.FIOUT-Datenblock !!!'
	GOTO 990
C       ----------------------------------------------------------------
990     CLOSE(10,STATUS='KEEP')
	CLOSE(11,STATUS='KEEP')
C	CLOSE(12,STATUS='KEEP')
	ERRFLG = 1
	RETURN
C     ------------------------------------------------------------------
C     Dateien schlieáen, (Datei "temp.FIOUT" l”schen).
C
1000  CLOSE(10,STATUS='KEEP')
      CLOSE(11,STATUS='KEEP')
C      CLOSE(12,STATUS='KEEP')
C      CALL SYSTEM('rm temp.FIOUT')
C
C     ------------------------------------------------------------------
C                                                      >>> P R I N T <<<
C      PRINT*,'beendet.'
      RETURN
      END
C
C=======================================================================
