C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE KELE(lnods,lnods_num,coord_num,
     *                nelem,nelem_max,nkd,
     *                kelem,kelem_adr,nl_kelem,
     *                help,npoin,myid,parallel)
c
      implicit none

      include 'mpif.h'

      integer   lnods,lnods_num,coord_num,
     *          nelem,nelem_max,nkd,
     *          kelem,kelem_adr,nl_kelem,
     *          help,npoin,myid

      integer   i,k,ielem,ipoin,adr,nnn,luerr,kelem_con,
     *          error_number,nnn_adr,nnn_mat,nerr,ludru,ierr

      logical   parallel,ausdrucken

      character*80 file_name,comment

      parameter (error_number=-99999,ludru=83)

      dimension lnods(nelem_max,nkd),lnods_num(nelem),
     *          coord_num(npoin),
     *          kelem(nl_kelem),kelem_adr(npoin+1),help(npoin)
c     ****************************************************************


c     ****************************************************************
c     KONTROLLE DER ELEMENT-KNOTEN-LISTE:

      nerr=0
      do 501 i=1,nelem
         do 502 k=1,nkd
            if (lnods(i,k).gt.npoin.or.lnods(i,k).le.0) then
               nerr=nerr+1
            endif
 502     continue
 501  continue

      if (nerr.ne.0) then
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine KELE'
         write(luerr,*)'Die Element-Knoten-Liste enthaelt unzulaessige'
         write(luerr,*)'Eintraege. Entweder sind die Eintraege kleiner'
         write(luerr,*)'gleich Null oder die Eintraege sind groesser  '
         write(luerr,*)'als die Knotenanzahl.                         '
         write(luerr,*)'Anzahl fehlerhafter Eintraege:',nerr 
         call erro_ende(myid,parallel,luerr)
      endif
c     ****************************************************************

c     ****************************************************************
c     INITIALISIERUNGEN:

      do 201 i=1,npoin+1
         kelem_adr(i)=error_number
 201  continue

      do 202 i=1,nl_kelem
         kelem(i)=error_number
 202  continue
c     ****************************************************************


c     ****************************************************************
c     BESTIMMUNG DER AN DEN KNOTEN BETEILIGTEN ELEMENTE:

c     Initialisierung:
      do 101 i=1,npoin
        help(i)=0
 101  continue

c     Bestimmung der Anzahl Elemente die an jedem Knoten beteiligt sind:
      do 100 ielem=1,nelem
         do 110 k=1,nkd
            ipoin=lnods(ielem,k)
            help(ipoin)=help(ipoin)+1
 110     continue         
 100  continue

c     Belegung des Adressfeldes:
      kelem_adr(1)=1
      kelem_con=0
      do 120 i=1,npoin
         kelem_con=MAX(kelem_con,help(i))
         kelem_adr(i+1)=kelem_adr(i)+help(i)
         help(i)=0
 120  continue

      nnn=kelem_adr(npoin+1)-kelem_adr(1)
      if (nnn.gt.nl_kelem) then
        call erro_init(myid,parallel,luerr)
        write(luerr,*)'Fehler in Routine KELE      '
        write(luerr,*)'Die Abschaetzung von Dimension nl_kelem'
        write(luerr,*)'im Hauptprogramm ist falsch.           '
        write(luerr,*)'Abgeschaetzte Dimension:',nl_kelem
        write(luerr,*)'Benoetigte    Dimension:',nnn
        call erro_ende(myid,parallel,luerr)
      endif

c     Bestimmung der Elemente die an jedem Knoten beteiligt sind:
      do 140 k=1,nkd
         do 130 ielem=1,nelem
            ipoin=lnods(ielem,k)
            adr=kelem_adr(ipoin)+help(ipoin)
            kelem(adr)=ielem
            help(ipoin)=help(ipoin)+1
 130     continue
 140  continue
c     ****************************************************************


c     ****************************************************************
c     KONTROLLE OB ALLE FELDER BESCHRIEBEN WURDEN: 

      nnn_adr=0
      nnn_mat=0
      do 701 i=1,npoin+1
         if (kelem_adr(i).eq.error_number) then
            nnn_adr=nnn_adr+1
         endif
 701  continue

      do 702 i=1,nl_kelem   
         if (kelem(i).eq.error_number) then
            nnn_mat=nnn_mat+1
         endif
 702  continue

      if (nnn_adr.ne.0.or.nnn_mat.ne.0) then
         call erro_init(myid,parallel,luerr)     
         write(luerr,*)'Fehler in Routine KELE               '
         write(luerr,*)'Die Felder zur Beschreibung der       '
         write(luerr,*)'Knoten-Element Verbindungen wurden       '
         write(luerr,*)'nicht vollstaendig beschriftet.          '
         write(luerr,*)'                                          '
         write(luerr,*)'Anzahl fehlender Eintraege in kelem_adr:', 
     *                     nnn_adr
         write(luerr,*)'Anzahl fehlender Eintraege in kelem    :',
     *                     nnn_mat 
         call erro_ende(myid,parallel,luerr)     
      endif
c     ****************************************************************


c     ****************************************************************
c     AUSDRUCK:

      ausdrucken=.false.
c     ausdrucken=.true.

      IF (ausdrucken) THEN

         if (parallel) then
            CALL MPI_BARRIER(MPI_COMM_WORLD,ierr)
         endif

         file_name='KELE'
         CALL FILE_OPEN(file_name,parallel,myid,ludru)

c        write(ludru,*)' Nr   lnods             '
c        do 580 i=1,nelem
c           write(ludru,566) lnods_num(i),
c    *            (coord_num(lnods(i,k)),k=1,nkd)
c580     continue

         write(ludru,*)' Nr     kelem '
         do 500 i=1,npoin
           write(ludru,555) coord_num(i),
     *                     (lnods_num(kelem(k)),
     *                k=kelem_adr(i),kelem_adr(i+1)-1)
 500     continue

         close(ludru)
         comment='File geschriebn:'
         call char_druck(comment,file_name,6)

         if (parallel) then
            CALL MPI_BARRIER(MPI_COMM_WORLD,ierr)
         endif
      ENDIF
 555  format(i6,3x,20(i6,1x))
 566  format(i6,1x,3x,8(i6,1x))
c     ****************************************************************

      return
      end

