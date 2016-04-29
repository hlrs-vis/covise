C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE GRAPH_KAN(lnods,coord_num,nelem,nelem_max,nkd,
     *                     kelem,kelem_adr,nl_kelem,
     *                     knmat,knmat_adr,nl_knmat,
     *                     help,zeig,npoin,
     *                     lnods_num,grad_max,grad_min,
     *                     myid,parallel,lupar,
     *                     redu_graph,schreiben)
c

      implicit none

      include 'mpif.h'

      integer   lnods,coord_num,nelem,nelem_max,nkd,
     *          kelem,kelem_adr,nl_kelem,
     *          knmat,knmat_adr,nl_knmat,
     *          help,zeig,npoin,
     *          lnods_num,grad_max,grad_min,
     *          myid

      integer   i,j,k,kn_num,ipoin,nnn,el_num,ze_num,nlang,
     *          idummy,kflag,luerr,error_number,ludru,
     *          nnn_adr,nnn_mat,nnn_err,ierr,lupar

      integer   kan_kn(8,3),nkan

      logical   schreiben,redu_graph,parallel,ausdrucken,
     *          diag_include

      character*80 file_name,comment

      parameter (error_number=-9999,ludru=71)

      dimension lnods(nelem_max,nkd),coord_num(npoin),
     *          lnods_num(nelem),
     *          kelem(nl_kelem),kelem_adr(npoin+1),
     *          help(npoin),zeig(npoin)

      dimension knmat(nl_knmat),knmat_adr(npoin+1)
c     ****************************************************************


c     ****************************************************************
c     INITIALISIERUNGEN:

      IF (schreiben) THEN
         do 101 i=1,npoin+1
            knmat_adr(i)=error_number
 101     continue

         do 102 j=1,nl_knmat    
            knmat(j)=error_number
 102     continue

         knmat_adr(1)=1
      ENDIF


      do 103 i=1,npoin
        help(i)=0
        zeig(i)=0
 103  continue

      nlang=0
c     ****************************************************************


c     ****************************************************************
c     BELEGUNG DES KANTENZEIGERS:

      if (nkd.eq.4) then

c       Inerhalb eines Elementes gehen von jedem Punkt 2 Kanten weg
        nkan=2

        kan_kn(1,1)=2
        kan_kn(1,2)=4

        kan_kn(2,1)=1
        kan_kn(2,2)=3

        kan_kn(3,1)=2
        kan_kn(3,2)=4

        kan_kn(4,1)=1
        kan_kn(4,2)=3

      else if (nkd.eq.8) then

c       Inerhalb eines Elementes gehen von jedem Punkt 3 Kanten weg
        nkan=3

        kan_kn(1,1)=2
        kan_kn(1,2)=4
        kan_kn(1,3)=5

        kan_kn(2,1)=1
        kan_kn(2,2)=3
        kan_kn(2,3)=6

        kan_kn(3,1)=2
        kan_kn(3,2)=4
        kan_kn(3,3)=7

        kan_kn(4,1)=1
        kan_kn(4,2)=3
        kan_kn(4,3)=8

        kan_kn(5,1)=1
        kan_kn(5,2)=6
        kan_kn(5,3)=8

        kan_kn(6,1)=2
        kan_kn(6,2)=5
        kan_kn(6,3)=7

        kan_kn(7,1)=3
        kan_kn(7,2)=6
        kan_kn(7,3)=8

        kan_kn(8,1)=4
        kan_kn(8,2)=5
        kan_kn(8,3)=7

      endif
c     *****************************************************************


c     ****************************************************************
c     BESTIMMUNG DES KNOTEN-GRAPHEN:

      diag_include=.false.

      do 300 ipoin=1,npoin

c       Bestimmung der Nachbarknoten von Knoten ipoin:
        if (diag_include) then
           nnn=1
           help(nnn)=ipoin
           zeig(ipoin)=1 
        else
           nnn=0
        endif

        do 310 i=kelem_adr(ipoin),kelem_adr(ipoin+1)-1
           el_num=kelem(i)
           do 320 k=1,nkd

              IF (redu_graph) THEN

c                Berechnung des reduzierten Graphen ueber die Kanten:
                 if (lnods(el_num,k).eq.ipoin) then
                    do 330 j=1,nkan
                       kn_num=lnods(el_num,kan_kn(k,j))
                       if (zeig(kn_num).eq.0.and.kn_num.ne.ipoin) then
c                         Knoten kn_num wurde noch nicht gezaehlt 
                          nnn=nnn+1
                          zeig(kn_num)=1
                          help(nnn)=kn_num
                       endif
 330                continue
                 endif

              ELSE 

c                Berechnung des gesamten Graphen:
                 kn_num=lnods(el_num,k)
                 if (zeig(kn_num).eq.0.and.kn_num.ne.ipoin) then
c                   Knoten kn_num wurde noch nicht gezaehlt 
                    nnn=nnn+1
                    zeig(kn_num)=1
                    help(nnn)=kn_num
                 endif

              ENDIF

 320       continue
 310    continue

        kflag=1
        call isort(help,idummy,nnn,kflag)

c       Beschriftung des Matrix-Graphen:  
        IF (schreiben) THEN

           do 340 i=1,nnn
             nlang=nlang+1
             knmat(nlang)=help(i)
             zeig(help(i))=0
 340       continue
           knmat_adr(ipoin+1)=knmat_adr(ipoin)+nnn

        ELSE

           do 350 i=1,nnn
             nlang=nlang+1
             zeig(help(i))=0
 350       continue

        ENDIF

 300  continue
c     ****************************************************************


c     ****************************************************************
c     DIMESNIONSKONTROLLE:

      IF (schreiben) THEN

         if (nlang.ne.nl_knmat) THEN
            call erro_init(myid,parallel,luerr)
            write(luerr,*)'Fehler in Routine GRAPH_NOD'
            write(luerr,*)'Die tatsaechlich geschriebene Anzahl an  '
            write(luerr,*)'an Matrixeintraegen stimmt mit der zuvor '
            write(luerr,*)'bestimmten Anzahl nicht ueberein.        '
            write(luerr,*)'Geschriebene Anzahl:',nlang           
            write(luerr,*)'Bestimmte    Anzahl:',nl_knmat  
            call erro_ende(myid,parallel,luerr)
         endif
      ELSE 

        nl_knmat=nlang

      ENDIF
c     ****************************************************************


c     ****************************************************************
c     KONTROLLE OB ALLE FELDER BESCHRIEBEN WURDEN: 

      IF (schreiben) THEN
         nnn_adr=0
         nnn_mat=0
         do 701 i=1,npoin+1
            if (knmat_adr(i).eq.error_number) then
               nnn_adr=nnn_adr+1
            endif
 701     continue

         do 702 i=1,nl_knmat  
            if (knmat(i).eq.error_number) then
               nnn_mat=nnn_mat+1
            endif
 702     continue

         if (nnn_adr.ne.0.or.nnn_mat.ne.0) then
            call erro_init(myid,parallel,luerr)     
            write(luerr,*)'Fehler in Routine GRAPH_NOD'
            write(luerr,*)'Die Felder zur Beschreibung des Graphen   '
            write(luerr,*)'wurden nicht vollstaendig beschriftet'
            write(luerr,*)'beschriftet.                              '
            write(luerr,*)'                                          '
            write(luerr,*)'Anzahl fehlender Eintraege in knmat_adr:',
     *                        nnn_adr
            write(luerr,*)'Anzahl fehlender Eintraege in knmat    :',
     *                        nnn_mat
            call erro_ende(myid,parallel,luerr)     
         endif
      ENDIF
c     ****************************************************************

      
c     ****************************************************************
c     KONTROLLE OB DIE MATRIX STRUKTURSYMMETRISCH IST:

      IF (schreiben) THEN

         nnn_err=0
         do 800 i=1,npoin
            do 810 k=knmat_adr(i),knmat_adr(i+1)-1
               ze_num=knmat(k)
c              Suchen in Zeile ze_num nach Spalte i:
               do 820 j=knmat_adr(ze_num),knmat_adr(ze_num+1)-1
                  if (knmat(j).eq.i) then
                     goto 821
                  endif
 820           continue           

               nnn_err=nnn_err+1

 821           continue
 810        continue
 800     continue

c        write(6,*)'Fehlermeldung GRAPH_NOD auskommentiert nnn_err=',
c    *                                                     nnn_err
c        nnn_err=0

         if (nnn_err.ne.0) then
            call erro_init(myid,parallel,luerr)
            write(luerr,*)'Fehler in Routine GRAPH_NOD'
            write(luerr,*)'Es wurden ',nnn_err,' Matrixeintraege     '
            write(luerr,*)'gefunden zu denen kein transponierter     '
            write(luerr,*)'Eintrag existiert. Fuer die Berechnung    '
            write(luerr,*)'muss die Matrix-Sturktur aber symmetrisch '
            write(luerr,*)'sein.                                     '
            call erro_ende(myid,parallel,luerr)
         endif
      ENDIF
c     ****************************************************************


c     ****************************************************************
c     BESTIMMUNG DES MINIMALEN UND MAXIMALEN GRADES:

      IF (schreiben) THEN
         grad_max=-1000000
         grad_min=+1000000
         do 600 i=1,npoin
            nnn=knmat_adr(i+1)-knmat_adr(i)
            grad_max=MAX(grad_max,nnn)
            grad_min=MIN(grad_min,nnn)
 600     continue
      ENDIF
c     ****************************************************************

      
c     ****************************************************************
c     AUSDRUCK DES MATRIX-GRAPHEN:

      ausdrucken=.false.
c     ausdrucken=.true. 

      IF (schreiben.and.ausdrucken) THEN

         if (parallel) then
            CALL MPI_BARRIER(MPI_COMM_WORLD,ierr)
         endif

         file_name='KNMAT'
         CALL FILE_OPEN(file_name,parallel,myid,ludru)
c 
         write(ludru,*)'AUSDRUCK AUS GRAPH_NOD nl_knmat=',nl_knmat  
         do 500 i=1,npoin

           nnn=0
           do 501 k=knmat_adr(i),knmat_adr(i+1)-1
              nnn=nnn+1
              help(nnn)=coord_num(knmat(k))
              kflag=1
              call isort(help,idummy,nnn,kflag)
 501       continue

           write(ludru,444) coord_num(i),knmat_adr(i+1)-knmat_adr(i),
     *                      (help(k),k=1,nnn)

c    *                 (coord_num(knmat(k)),
c    *                k=knmat_adr(i),knmat_adr(i+1)-1)

 500     continue

         close(ludru)
         comment='File geschriebn:'
         call char_druck(comment,file_name,6)

         if (parallel) then
            CALL MPI_BARRIER(MPI_COMM_WORLD,ierr)
         endif

      ENDIF
 444  format(i6,1x,i3,1x,50(i5,1x))
c444  format(i2,2x,i2,3x,50(i2,1x))
c     ****************************************************************


      return
      end
