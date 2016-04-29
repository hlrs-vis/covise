C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
C *****************                         ********************
C *****************   Version 3.4.3 (2/94)  ********************
C *****************                         ********************
C **************************************************************
C
C
C
C Unterprogramme   char   
C                  isort      
C                  ssort      
C                  icopy      
C                  scopy      
C                  ARC_COS     
C                  char_druck
C                  erro_init
C                  erro_ende
C                  FILE_OPEN
C                  ALLOC_ENDE
C                  ALLOC_FINE
C                  DEALLOC_ALLE
C                  SPEICH_CHECK


C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      function lentb(cs)
c
c bestimmen der Laenge des Characterstrings ohne Trailing Blanks
c
      implicit none
      integer i,lentb
      character*(*) cs
      do 5 i=len(cs),1,-1
          if(cs(i:i).ne.' ') goto 10
5     continue
10    lentb=i
      return
      end
c
c
      subroutine cbl(st)
c
c Belegen des Strings ST mit BLANK
c
      implicit none
c
      character*(*) st
c
      integer i,len
c
c
      do 5 i=1,len(st)
          st(i:i)=' '
5     continue
      return
      end
c
c
c
      subroutine stcaps(cs)
c
c  Ersetzen von Kleinbuchstaben durch Grossbuchstaben im
c  Characterstring cs
c
      implicit none
c
      character*(*) cs
      character*1 char
      integer n,ichar,i,j,len
c
      n=len(cs)
      do 5 i=1,n
           j=ichar(cs(i:i))
           if(j.ge.97.and.j.le.122) then
                cs(i:i)=char(j-32)
           end if
5     continue
      return
      end
c
c
c
      subroutine stsmal(cs)
c
c  Ersetzen von Grossbuchstaben durch Kleinbuchstaben im
c  Characterstring cs
c
      implicit none
c
      character*(*) cs
      character*1 char
      integer n,ichar,i,j,len
c
      n=len(cs)
      do 5 i=1,n
           j=ichar(cs(i:i))
           if(j.ge.65.and.j.le.90) then
                cs(i:i)=char(j+32)
           end if
5     continue
      return
      end
c
c
c
      function lenfb(cs)
c
c ausscheiden fuehrender Blanks
c
      implicit none
c
      integer lo,i,lf,lenfb,lentb
c
      character*(*) cs
c
      external lentb
c
      lo=len(cs)
      do 10 i=1,lo
           if(cs(i:i).ne.' ') goto 20
10    continue
      lenfb=0
      return
c
c
20    continue
      lf=i-1
      if(lf.ne.0) then
           do 30 i=1,lo-lf
                cs(i:i)=cs(i+lf:i+lf)
30         continue
           do 40 i=lo-lf+1,lo
                cs(i:i)=' '
40         continue
      end if
      lenfb=lentb(cs)
      return
      end
C
C
      SUBROUTINE DPREST(TE1,TE2)
C
C AUSSCHEIDEN EINES KOMMENTARS AM ANFANG EINER ZEILE,
C VERSCHIEBEN DES ZEILENINHALTS GANZ NACH LINKS
C BELEGEN DER RESTLICHEN ZEILE MIT BLANKS
C
C KOMMENTARENDE IST DURCH DOPPELPUNKT : (58) GEKENNZEICHNET.
c
      implicit none
C
      CHARACTER*(*) TE1,TE2
      INTEGER L1,L2,I,lentb,len,lenfb
      EXTERNAL LENFB,LENTB
      L1=LENTB(TE1)
      L2=LEN(TE2)
      IF(L1.EQ.0) THEN
            TE2=TE1
      ELSE
            DO 10 I=1,L1
                  IF(TE1(I:I).EQ.':') GOTO 15
10          CONTINUE
            TE2=TE1
            GOTO 20
15          CONTINUE
            if(i.eq.l1) then
            te2(1:l2)=' '
            else
            TE2(1:1+L1-I)=TE1(I+1:L1)
            TE2(L1-I+2:L2)=' '
            end if
20          CONTINUE
      END IF
      L2=LENFB(TE2)
      RETURN
      END
c
c
c
        function iread(string)
c
        implicit none
        include 'mpif.h'
        character*(*) string
        character*15 hs,hs2
        integer iread,in,i,l,lenfb,j,lentb,ierr
        external lenfb,lentb
c
        l=lentb(string)
        do 5 i=1,l
           if(string(i:i).eq.',') string(i:i)=' '
5       continue
c
        l=lenfb(string)
c
        if(l.eq.0) then
           iread=0.
           return
        end if
c
        j=0
        do 10 i=1,l
        if(string(i:i).eq.' ') goto 20
        j=j+1
        hs(j:j)=string(i:i)
        string(i:i) = ' '
10      continue
20      continue        
        do 30 i=1,j
        hs2(15-j+i:15-j+i)=hs(i:i)
30      continue
        do 40 i=1,15-j
        hs2(i:i)=' '
40      continue
        read(hs2,'(i15)',err=60) in
        iread=in
        return
60      write(*,*) '******************************'
        write(*,*) '** Syntax Error in IREAD *****'
        write(*,*) '******************************'
        write(*,*) hs2
        CALL MPI_ABORT(MPI_COMM_WORLD,MPI_ERR_UNKNOWN,ierr)
        stop
        end
c
c
c
        function rread(string)
c
        implicit none
        include 'mpif.h'
c
        integer  i,j,l,lenfb,lentb,ierr
c
        real wert,rread
c
        character*(*) string
        character*16 hs,hs2
c
        external lenfb,lentb
c
        l=lentb(string)
        do 5 i=1,l
           if(string(i:i).eq.',') string(i:i)=' '
5       continue
c
        l=lenfb(string)
        if(l.eq.0) then
            rread=0.
            return
        end if
c
        j=0
        do 10 i=1,l
           if(string(i:i).eq.' ') goto 20
           j=j+1
           hs(j:j)=string(i:i)
           string(i:i)=' '
10      continue
20      continue        
        do 30 i=1,j
                hs2(16-j+i:16-j+i)=hs(i:i)
30      continue
        do 40 i=1,16-j
                hs2(i:i)=' '
40      continue
                read(hs2,'(g16.0)',err=60) wert
        rread=wert
        return
60      write(*,*) '******************************'
        write(*,*) '** Syntax Error in RREAD *****'
        write(*,*) '******************************'
        write(*,*) hs2
        CALL MPI_ABORT(MPI_COMM_WORLD,MPI_ERR_UNKNOWN,ierr)
        stop
        end


C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE ISORT(X,Y,N,KFLAG)

      include 'common.zer'

      integer  luerr

      dimension IL(21),IU(21)
      integer X(N),Y(N),T,TT,TY,TTY
c     *****************************************************************

c     Description of Parameters
c         X - integer array of values to be sorted
c         Y - integer array to be (optionally) carried along
c         N - number of values in integer array X to be sorted
c     KFLAG - control parameter
c           = 2 means sort X in INCREASING order and carry Y along.
c           = 1 means sort X in INCREASING order (ignoring Y)
c           =-1 means sort X in DECREASING order (ignoring Y)
c           =-2 means sort X in DECREASING order and carry Y along.

C     *****************************************************************
C     FEHLER-MELDUNGEN:

      NN = N
      IF (NN.LT.0) THEN
        call erro_init(myid,parallel,luerr) 
        write(luerr,*)'Fehler in Routine isort'
        write(luerr,*)'Die Dimension N ist kleiner Null '
        write(luerr,*)'N=',N    
        call erro_ende(myid,parallel,luerr) 
      ENDIF
 
      IF (NN.EQ.0) THEN
         return
      ENDIF

      KK = IABS(KFLAG)
      IF ((KK.EQ.1).OR.(KK.EQ.2)) THEN      
         goto 15 
      ELSE
        call erro_init(myid,parallel,luerr) 
        write(luerr,*)'Fehler in Routine isort'
        write(luerr,*)'Der Parameter kflag besitzt einen ungueltigen'
        write(luerr,*)'Wert.'
        write(luerr,*)'Zulaessige   Werte: -2 / -1 / 1 / 2 '
        write(luerr,*)'Eingegebener Werte:',kflag             
        call erro_ende(myid,parallel,luerr) 
      ENDIF
C     *****************************************************************


 
C     *****************************************************************
C     ALTER ARRAY X TO GET DECREASING ORDER IF NEEDED
C
   15 IF (KFLAG.GE.1) GO TO 30
      DO 20 I=1,NN
   20 X(I) = -X(I)
   30 GO TO (100,200),KK
C
C SORT X ONLY
C
  100 CONTINUE
      M=1
      I=1
      J=NN
      R=.375
  110 IF (I .EQ. J) GO TO 155
  115 IF (R .GT. .5898437) GO TO 120
      R=R+3.90625E-2
      GO TO 125
  120 R=R-.21875
  125 K=I
C                                  SELECT A CENTRAL ELEMENT OF THE
C                                  ARRAY AND SAVE IT IN LOCATION T
      IJ = I + IFIX (FLOAT (J-I) * R)
      T=X(IJ)
C                                  IF FIRST ELEMENT OF ARRAY IS GREATER
C                                  THAN T, INTERCHANGE WITH T
      IF (X(I) .LE. T) GO TO 130
      X(IJ)=X(I)
      X(I)=T
      T=X(IJ)
  130 L=J
C                                  IF LAST ELEMENT OF ARRAY IS LESS THAN
C                                  T, INTERCHANGE WITH T
      IF (X(J) .GE. T) GO TO 140
      X(IJ)=X(J)
      X(J)=T
      T=X(IJ)
C                                  IF FIRST ELEMENT OF ARRAY IS GREATER
C                                  THAN T, INTERCHANGE WITH T
      IF (X(I) .LE. T) GO TO 140
      X(IJ)=X(I)
      X(I)=T
      T=X(IJ)
      GO TO 140
  135 TT=X(L)
      X(L)=X(K)
      X(K)=TT
C                                  FIND AN ELEMENT IN THE SECOND HALF OF
C                                  THE ARRAY WHICH IS SMALLER THAN T
  140 L=L-1
      IF (X(L) .GT. T) GO TO 140
C                                  FIND AN ELEMENT IN THE FIRST HALF OF
C                                  THE ARRAY WHICH IS GREATER THAN T
  145 K=K+1
      IF (X(K) .LT. T) GO TO 145
C                                  INTERCHANGE THESE ELEMENTS
      IF (K .LE. L) GO TO 135
C                                  SAVE UPPER AND LOWER SUBSCRIPTS OF
C                                  THE ARRAY YET TO BE SORTED
      IF (L-I .LE. J-K) GO TO 150
      IL(M)=I
      IU(M)=L
      I=K
      M=M+1
      GO TO 160
  150 IL(M)=K
      IU(M)=J
      J=L
      M=M+1
      GO TO 160
C                                  BEGIN AGAIN ON ANOTHER PORTION OF
C                                  THE UNSORTED ARRAY
  155 M=M-1
      IF (M .EQ. 0) GO TO 300
      I=IL(M)
      J=IU(M)
  160 IF (J-I .GE. 1) GO TO 125
      IF (I .EQ. 1) GO TO 110
      I=I-1
  165 I=I+1
      IF (I .EQ. J) GO TO 155
      T=X(I+1)
      IF (X(I) .LE. T) GO TO 165
      K=I
  170 X(K+1)=X(K)
      K=K-1
      IF (T .LT. X(K)) GO TO 170
      X(K+1)=T
      GO TO 165
C
C SORT X AND CARRY Y ALONG
C
  200 CONTINUE
      M=1
      I=1
      J=NN
      R=.375
  210 IF (I .EQ. J) GO TO 255
  215 IF (R .GT. .5898437) GO TO 220
      R=R+3.90625E-2
      GO TO 225
  220 R=R-.21875
  225 K=I
C                                  SELECT A CENTRAL ELEMENT OF THE
C                                  ARRAY AND SAVE IT IN LOCATION T
      IJ = I + IFIX (FLOAT (J-I) *R)
      T=X(IJ)
      TY= Y(IJ)
C                                  IF FIRST ELEMENT OF ARRAY IS GREATER
C                                  THAN T, INTERCHANGE WITH T
      IF (X(I) .LE. T) GO TO 230
      X(IJ)=X(I)
      X(I)=T
      T=X(IJ)
       Y(IJ)= Y(I)
       Y(I)=TY
      TY= Y(IJ)
  230 L=J
C                                  IF LAST ELEMENT OF ARRAY IS LESS THAN
C                                  T, INTERCHANGE WITH T
      IF (X(J) .GE. T) GO TO 240
      X(IJ)=X(J)
      X(J)=T
      T=X(IJ)
       Y(IJ)= Y(J)
       Y(J)=TY
      TY= Y(IJ)
C                                  IF FIRST ELEMENT OF ARRAY IS GREATER
C                                  THAN T, INTERCHANGE WITH T
      IF (X(I) .LE. T) GO TO 240
      X(IJ)=X(I)
      X(I)=T
      T=X(IJ)
       Y(IJ)= Y(I)
       Y(I)=TY
      TY= Y(IJ)
      GO TO 240
  235 TT=X(L)
      X(L)=X(K)
      X(K)=TT
      TTY= Y(L)
       Y(L)= Y(K)
       Y(K)=TTY
C                                  FIND AN ELEMENT IN THE SECOND HALF OF
C                                  THE ARRAY WHICH IS SMALLER THAN T
  240 L=L-1
      IF (X(L) .GT. T) GO TO 240
C                                  FIND AN ELEMENT IN THE FIRST HALF OF
C                                  THE ARRAY WHICH IS GREATER THAN T
  245 K=K+1
      IF (X(K) .LT. T) GO TO 245
C                                  INTERCHANGE THESE ELEMENTS
      IF (K .LE. L) GO TO 235
C                                  SAVE UPPER AND LOWER SUBSCRIPTS OF
C                                  THE ARRAY YET TO BE SORTED
      IF (L-I .LE. J-K) GO TO 250
      IL(M)=I
      IU(M)=L
      I=K
      M=M+1
      GO TO 260
  250 IL(M)=K
      IU(M)=J
      J=L
      M=M+1
      GO TO 260
C                                  BEGIN AGAIN ON ANOTHER PORTION OF
C                                  THE UNSORTED ARRAY
  255 M=M-1
      IF (M .EQ. 0) GO TO 300
      I=IL(M)
      J=IU(M)
  260 IF (J-I .GE. 1) GO TO 225
      IF (I .EQ. 1) GO TO 210
      I=I-1
  265 I=I+1
      IF (I .EQ. J) GO TO 255
      T=X(I+1)
      TY= Y(I+1)
      IF (X(I) .LE. T) GO TO 265
      K=I
  270 X(K+1)=X(K)
       Y(K+1)= Y(K)
      K=K-1
      IF (T .LT. X(K)) GO TO 270
      X(K+1)=T
       Y(K+1)=TY
      GO TO 265
C
C CLEAN UP
C
  300 IF (KFLAG.GE.1) RETURN
      DO 310 I=1,NN
  310 X(I) = -X(I)

c     *****************************************************************
c     Written by Rondall E Jones
c     Modified by John A. Wisniewski to use the Singleton QUICKSORT
c     algorithm. Date 18 November 1976.
c
c     Further modified by David K. Kahaner
c     NATIONAL BUREAU OF STANDARDS
c     August, 1981
c
c     References: SINGLETON, R. C., ALGORITHM 347, AN EFFICIENT
c                 ALGORITHM FOR SORTING WITH MINIMAL STORAGE, CACM,
c                 VOL. 12, NO. 3, 1969, PP. 185-187.
c     *****************************************************************
      RETURN
      END


C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE SSORT(X,Y,N,KFLAG)

      include 'common.zer'

      integer luerr

      dimension X(N),Y(N),IL(21),IU(21)
C     *****************************************************************


c     Description of Parameters
c         X - integer array of values to be sorted
c         Y - integer array to be (optionally) carried along
c         N - number of values in integer array X to be sorted
c     KFLAG - control parameter
c           = 2 means sort X in INCREASING order and carry Y along.
c           = 1 means sort X in INCREASING order (ignoring Y)
c           =-1 means sort X in DECREASING order (ignoring Y)
c           =-2 means sort X in DECREASING order and carry Y along.


c     *****************************************************************
c     FEHLER-MELDUNGEN:

      NN = N
      IF (NN.LT.0) THEN
        call erro_init(myid,parallel,luerr) 
        write(luerr,*)'Fehler in Routine ssort'
        write(luerr,*)'Die Dimension N ist kleiner Null '
        write(luerr,*)'N=',N    
        call erro_ende(myid,parallel,luerr) 
      ENDIF

      IF (NN.EQ.0) THEN
         return
      ENDIF

      KK = IABS(KFLAG)
      IF ((KK.EQ.1).OR.(KK.EQ.2)) THEN      
         goto 15 
      ELSE
        call erro_init(myid,parallel,luerr) 
        write(luerr,*)'Fehler in Routine ssort'
        write(luerr,*)'Der Parameter kflag besitzt einen ungueltigen '
        write(luerr,*)'Wert.'
        write(luerr,*)'Zulaessige   Werte: -2 / -1 / 1 / 2 '
        write(luerr,*)'Eingegebener Werte:',kflag             
        call erro_ende(myid,parallel,luerr) 
      ENDIF
C     *****************************************************************

 
C     *****************************************************************
C     ALTER ARRAY X TO GET DECREASING ORDER IF NEEDED
C
   15 IF (KFLAG.GE.1) GO TO 30
      DO 20 I=1,NN
   20 X(I) = -X(I)
   30 GO TO (100,200),KK
C
C SORT X ONLY
C
  100 CONTINUE
      M=1
      I=1
      J=NN
      R=.375
  110 IF (I .EQ. J) GO TO 155
  115 IF (R .GT. .5898437) GO TO 120
      R=R+3.90625E-2
      GO TO 125
  120 R=R-.21875
  125 K=I
C                                  SELECT A CENTRAL ELEMENT OF THE
C                                  ARRAY AND SAVE IT IN LOCATION T
      IJ = I + IFIX (FLOAT (J-I) * R)
      T=X(IJ)
C                                  IF FIRST ELEMENT OF ARRAY IS GREATER
C                                  THAN T, INTERCHANGE WITH T
      IF (X(I) .LE. T) GO TO 130
      X(IJ)=X(I)
      X(I)=T
      T=X(IJ)
  130 L=J
C                                  IF LAST ELEMENT OF ARRAY IS LESS THAN
C                                  T, INTERCHANGE WITH T
      IF (X(J) .GE. T) GO TO 140
      X(IJ)=X(J)
      X(J)=T
      T=X(IJ)
C                                  IF FIRST ELEMENT OF ARRAY IS GREATER
C                                  THAN T, INTERCHANGE WITH T
      IF (X(I) .LE. T) GO TO 140
      X(IJ)=X(I)
      X(I)=T
      T=X(IJ)
      GO TO 140
  135 TT=X(L)
      X(L)=X(K)
      X(K)=TT
C                                  FIND AN ELEMENT IN THE SECOND HALF OF
C                                  THE ARRAY WHICH IS SMALLER THAN T
  140 L=L-1
      IF (X(L) .GT. T) GO TO 140
C                                  FIND AN ELEMENT IN THE FIRST HALF OF
C                                  THE ARRAY WHICH IS GREATER THAN T
  145 K=K+1
      IF (X(K) .LT. T) GO TO 145
C                                  INTERCHANGE THESE ELEMENTS
      IF (K .LE. L) GO TO 135
C                                  SAVE UPPER AND LOWER SUBSCRIPTS OF
C                                  THE ARRAY YET TO BE SORTED
      IF (L-I .LE. J-K) GO TO 150
      IL(M)=I
      IU(M)=L
      I=K
      M=M+1
      GO TO 160
  150 IL(M)=K
      IU(M)=J
      J=L
      M=M+1
      GO TO 160
C                                  BEGIN AGAIN ON ANOTHER PORTION OF
C                                  THE UNSORTED ARRAY
  155 M=M-1
      IF (M .EQ. 0) GO TO 300
      I=IL(M)
      J=IU(M)
  160 IF (J-I .GE. 1) GO TO 125
      IF (I .EQ. 1) GO TO 110
      I=I-1
  165 I=I+1
      IF (I .EQ. J) GO TO 155
      T=X(I+1)
      IF (X(I) .LE. T) GO TO 165
      K=I
  170 X(K+1)=X(K)
      K=K-1
      IF (T .LT. X(K)) GO TO 170
      X(K+1)=T
      GO TO 165
C
C SORT X AND CARRY Y ALONG
C
  200 CONTINUE
      M=1
      I=1
      J=NN
      R=.375
  210 IF (I .EQ. J) GO TO 255
  215 IF (R .GT. .5898437) GO TO 220
      R=R+3.90625E-2
      GO TO 225
  220 R=R-.21875
  225 K=I
C                                  SELECT A CENTRAL ELEMENT OF THE
C                                  ARRAY AND SAVE IT IN LOCATION T
      IJ = I + IFIX (FLOAT (J-I) *R)
      T=X(IJ)
      TY= Y(IJ)
C                                  IF FIRST ELEMENT OF ARRAY IS GREATER
C                                  THAN T, INTERCHANGE WITH T
      IF (X(I) .LE. T) GO TO 230
      X(IJ)=X(I)
      X(I)=T
      T=X(IJ)
       Y(IJ)= Y(I)
       Y(I)=TY
      TY= Y(IJ)
  230 L=J
C                                  IF LAST ELEMENT OF ARRAY IS LESS THAN
C                                  T, INTERCHANGE WITH T
      IF (X(J) .GE. T) GO TO 240
      X(IJ)=X(J)
      X(J)=T
      T=X(IJ)
       Y(IJ)= Y(J)
       Y(J)=TY
      TY= Y(IJ)
C                                  IF FIRST ELEMENT OF ARRAY IS GREATER
C                                  THAN T, INTERCHANGE WITH T
      IF (X(I) .LE. T) GO TO 240
      X(IJ)=X(I)
      X(I)=T
      T=X(IJ)
       Y(IJ)= Y(I)
       Y(I)=TY
      TY= Y(IJ)
      GO TO 240
  235 TT=X(L)
      X(L)=X(K)
      X(K)=TT
      TTY= Y(L)
       Y(L)= Y(K)
       Y(K)=TTY
C                                  FIND AN ELEMENT IN THE SECOND HALF OF
C                                  THE ARRAY WHICH IS SMALLER THAN T
  240 L=L-1
      IF (X(L) .GT. T) GO TO 240
C                                  FIND AN ELEMENT IN THE FIRST HALF OF
C                                  THE ARRAY WHICH IS GREATER THAN T
  245 K=K+1
      IF (X(K) .LT. T) GO TO 245
C                                  INTERCHANGE THESE ELEMENTS
      IF (K .LE. L) GO TO 235
C                                  SAVE UPPER AND LOWER SUBSCRIPTS OF
C                                  THE ARRAY YET TO BE SORTED
      IF (L-I .LE. J-K) GO TO 250
      IL(M)=I
      IU(M)=L
      I=K
      M=M+1
      GO TO 260
  250 IL(M)=K
      IU(M)=J
      J=L
      M=M+1
      GO TO 260
C                                  BEGIN AGAIN ON ANOTHER PORTION OF
C                                  THE UNSORTED ARRAY
  255 M=M-1
      IF (M .EQ. 0) GO TO 300
      I=IL(M)
      J=IU(M)
  260 IF (J-I .GE. 1) GO TO 225
      IF (I .EQ. 1) GO TO 210
      I=I-1
  265 I=I+1
      IF (I .EQ. J) GO TO 255
      T=X(I+1)
      TY= Y(I+1)
      IF (X(I) .LE. T) GO TO 265
      K=I
  270 X(K+1)=X(K)
       Y(K+1)= Y(K)
      K=K-1
      IF (T .LT. X(K)) GO TO 270
      X(K+1)=T
       Y(K+1)=TY
      GO TO 265
C
C CLEAN UP
C
  300 IF (KFLAG.GE.1) RETURN
      DO 310 I=1,NN
  310 X(I) = -X(I)

c     *****************************************************************
c     Written by Rondall E Jones
c     Modified by John A. Wisniewski to use the Singleton QUICKSORT
c     algorithm. Date 18 November 1976.
c
c     Further modified by David K. Kahaner
c     NATIONAL BUREAU OF STANDARDS
c     August, 1981
c
c     References: SINGLETON, R. C., ALGORITHM 347, AN EFFICIENT
c                 ALGORITHM FOR SORTING WITH MINIMAL STORAGE, CACM,
c                 VOL. 12, NO. 3, 1969, PP. 185-187.
c     *****************************************************************
      RETURN
      END


C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE scopy(feld_a,feld_b,nnn)        
      implicit none
      integer   nnn,i
      real      feld_a,feld_b
      dimension feld_a(nnn),feld_b(nnn)
      do 55 i=1,nnn   
           feld_b(i)=feld_a(i)
 55   continue
      return
      end 


C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE icopy(feld_a,feld_b,nnn)        
      implicit none
      integer   nnn,i
      integer   feld_a,feld_b
      dimension feld_a(nnn),feld_b(nnn)
      do 55 i=1,nnn   
           feld_b(i)=feld_a(i)
 55   continue
      return
      end 

C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE inte_init(feld,nnn,number)        
      implicit none
      integer   nnn,i,number
      integer   feld
      dimension feld(nnn)
      do 55 i=1,nnn   
           feld(i)=number       
 55   continue
      return
      end 

C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE real_init(feld,nnn,wert)        
      implicit none
      integer   nnn,i
      real      feld,wert
      dimension feld(nnn)
      do 55 i=1,nnn   
           feld(i)=wert
 55   continue
      return
      end 

C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE ARC_COS(wert,wink,alfa,myid,parallel)

      implicit none 

      integer  luerr,myid

      logical  parallel
      
      real     wert,wink,alfa,pi,tol_abs,ddd_0,ddd_1,ddd_2

      parameter (pi=3.141592654,tol_abs=5.e-05)
c     ****************************************************************


c     ****************************************************************
c     BESTIMMUNG DER TOLERANZ:
 
      ddd_0=abs(wert)
      ddd_1=abs(1.0-wert)
      ddd_2=abs(1.0+wert)

      if (ddd_0.lt.tol_abs) then

c        Fall: wert=0.0
         alfa=90.0
         wink=pi/2.0

      else if (ddd_1.lt.tol_abs) then

c        Fall: wert=1.0
         alfa=0.0
         wink=0.0      

      else if (ddd_2.lt.tol_abs) then

c        Fall: wert=-1.0
         alfa=180.0
         wink=pi       

      else 
      
          if (wert.gt.1.0.or.wert.lt.-1.0) then
             call erro_init(myid,parallel,luerr)
             write(luerr,*)'Fehler in Routine ARC_COS'
             write(luerr,*)'Das Argument der Arkus-Kosinus-Funktion'
             write(luerr,*)'muss zwischen 1 und -1 liegen          '
             write(luerr,*)'Argument:',wert                        
             call erro_ende(myid,parallel,luerr)
          endif

          wink=ACOS(wert) 
          alfa=wink*180.0/pi

      endif
c     ****************************************************************

      return
      end



C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      subroutine char_druck(name_1,name_2,lu)
      implicit none
      integer ilang_1,ilang_2,lu,lentb
      character*80 name_1,name_2
      character*2 otto
      character*15 forma

      ilang_1=lentb(name_1)
      ilang_2=lentb(name_2)
      forma='(1x,A  ,1x,A  )'

      write(otto,'(i2.2)') ilang_1
      forma(6:7)=otto(1:2)

      write(otto,'(i2.2)') ilang_2
      forma(13:14)=otto(1:2)

      write(lu,forma) name_1,name_2

      return
      end



C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE erro_init(myid,parallel,luerr)
      implicit none
      include 'mpif.h'
      integer  myid,luerr
      logical  parallel
      character*80 err_file
      character*3  otto      

      if (parallel) then
        luerr=99
        write(otto,'(i3.3)') myid+1
        err_file='ERROR_'
        err_file(7:9)=otto(1:3)
        open(luerr,file=err_file,status='unknown')
        write(6,*) '               '
        write(6,*) 'FEHLER AUF PROZESSOR ',myid
      else 
        luerr=6             
        write(6,*) '               '
      endif

      return
      end


C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE erro_ende(myid,parallel,luerr)
      implicit none
      include 'mpif.h'
      integer  myid,luerr,ierr,ierrcode
      logical  parallel
      character*80 err_file
      character*80 comment 
      character*3  otto      

      if (parallel) then
        ierrcode=MPI_ERR_UNKNOWN
        write(otto,'(i3.3)') myid+1
        err_file='ERROR_'
        err_file(7:9)=otto(1:3)
        close(luerr)
        comment='Name der Error-Files:'
        call char_druck(comment,err_file,6)
        write(6,*) 'PROGRAMM-ABBRUCH !!!!!!!!!!             '
        write(6,*) '               '
        CALL MPI_ABORT(MPI_COMM_WORLD,ierrcode,ierr)
        stop
      else 
        write(6,*) 'PROGRAMM-ABBRUCH !!!!!!!!!!             '
        write(6,*) '               '
        stop                
      endif

      return
      end


C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE FILE_OPEN(para_name,parallelo,myid_num,lufil)
      implicit none

      include 'common.zer'
      
      integer lentb,ilang,myid_num,lufil
      logical parallelo

      character*80 para_name
      character*2  otto_1,otto_2,otto_3
      character*3  otto_4
c     ****************************************************************

c     ****************************************************************
c     write(otto_1,'(i2.2)') iglo_mom
c     write(otto_2,'(i2.2)') izwi_mom
c     write(otto_3,'(i2.2)') igln_mom
c     write(otto_4,'(i3.3)') myid_num+1

      ilang=lentb(para_name)

c     para_name(ilang+1:ilang+1)='_'           
c     para_name(ilang+2:ilang+3)=otto_1(1:2)
c     para_name(ilang+4:ilang+4)='_'           
c     para_name(ilang+5:ilang+6)=otto_2(1:2)
c     para_name(ilang+7:ilang+7)='_'           
c     para_name(ilang+8:ilang+9)=otto_3(1:2)
c     if (parallelo) then
c        para_name(ilang+10:ilang+10)='.'           
c        para_name(ilang+11:ilang+13)=otto_4(1:3)   
c     endif

      if (parallelo) then
        para_name(ilang+1:ilang+1)='_'
        para_name(ilang+2:ilang+4)=otto_4(1:3)
      endif

      
      open(lufil,file=para_name,status='unknown')
c     ****************************************************************

      return
      end




C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE ALLOC_ENDE(iende,iname,nnn)
      implicit none
      integer iende,iname,nnn
      iname=iende
      iende=iname+nnn
      return
      end

C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE ALLOC_FINE(ifine,iname,nnn)
      implicit none
      integer ifine,iname,nnn
      iname=ifine-nnn
      ifine=iname
      return
      end

C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE DEALLOC_ALLE(ifine,lmax)
      implicit none
      integer ifine,lmax           
      ifine=lmax          
      return
      end


C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE SPEICH_CHECK(iende,ifine,lmax,speich_max)
      implicit none
      include 'mpif.h'
      include 'common.zer'
      integer iende,ifine,lmax,speich_max,speich_mom,luerr
      if (iende.ge.ifine) then
        call erro_init(myid,parallel,luerr)
        write(luerr,*) 'SPEICHER ZU KLEIN !!!!                       '
        write(luerr,*) 'Es fehlen ',ABS(iende-ifine),' Speicherplaetze'
        write(luerr,*) 'lmax =',lmax     
        write(luerr,*) 'iende=',iende
        write(luerr,*) 'ifine=',ifine
        call erro_ende(myid,parallel,luerr)
      endif
      speich_mom=iende+(lmax-ifine)
      speich_max=MAX(speich_max,speich_mom)
      
      return
      end
      
