C **************************************************************
c **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE ERG_LES(coord_num,lnods_num,
     *                   erg_kn,nsp_kn_erg,
     *                   erg_el,nsp_el_erg,
     *                   erg_za,nsp_za_erg,
     *                   transi_erg,erg_name)
  
      implicit none     

      include 'mpif.h'
      include 'common.zer'

      integer coord_num,lnods_num,iread,
     *        nsp_kn_erg,nsp_el_erg,nsp_za_erg,
     *        npoin_erg,nelem_erg,npoin_zal

      integer i,j,lu,luerr,kn_num,el_num

      real    erg_kn,erg_el,erg_za

      character*80 erg_name,comment,reihe,file_name

      logical  erro_print,transi_erg,format_read

      parameter (lu=97)
C
      dimension lnods_num(nelem_max),coord_num(npoin_max),
     *          erg_kn(npoin_max,nsp_kn_erg),
     *          erg_el(nelem_max,nsp_el_erg),
     *          erg_za(npoin_max,nsp_za_erg)
c     *****************************************************************


c     **********************************************************
c     OEFFNEN DER FILES UND DIMENSIONSKONTROLLE:

      npoin_erg=0
      npoin_zal=0
      nelem_erg=0

      file_name=erg_name
      open(lu,file=erg_name,status='old',err=777)
      format_read=.true.
      CALL HEAD_READ(lu,file_name,format_read,reihe)

      npoin_erg=iread(reihe)
      nelem_erg=iread(reihe)
      npoin_zal=iread(reihe)

      erro_print=.false.        
      if (npoin.ne.npoin_erg.or.nelem.ne.nelem_erg) then
         erro_print=.true.
      endif
      if (npoin_zal.ne.0.and.npoin_zal.ne.npoin) then
         erro_print=.true.
      endif

      if (erro_print) then
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine ERG_LES'
         write(luerr,*)'Die Dimensionen im Geometrie-File und'
         write(luerr,*)'Ergebnis-File sind nicht konsistent. '
         write(luerr,*)'npoin_geo=',npoin
         write(luerr,*)'npoin_erg=',npoin_erg
         if (npoin_zal.ne.0) then
         write(luerr,*)'npoin_zal=',npoin_zal
         endif
         write(luerr,*)'nelem_geo=',nelem
         write(luerr,*)'nelem_erg=',nelem_erg
         call erro_ende(myid,parallel,luerr)
      endif

      transi_erg=.false.
      if (npoin_zal.eq.npoin) then
         transi_erg=.true.
      endif
c     **********************************************************


c     **********************************************************
c     EINLESEN DER ERGEBISSE:

      do 30 i=1,npoin
           read(lu,*) kn_num,(erg_kn(i,j),j=1,nsp_kn_erg)
           if (kn_num.ne.coord_num(i)) THEN
              call erro_init(myid,parallel,luerr)
              write(luerr,*)'FEHLER IM ERGEBNISFILE                '
              write(luerr,*)'Die Knotennumerierung im Ergebnis-File '
              write(luerr,*)'und im Geometrie-File ist unterschiedlich.'
              write(luerr,*)'Adresse                 :',i      
              write(luerr,*)'Nummer im Geometrie-File:',coord_num(i)
              write(luerr,*)'Nummer im Ergebnis -File:',kn_num      
              call erro_ende(myid,parallel,luerr)
           endif
 30   continue

      do 40 i=1,nelem
           read(lu,*) el_num,(erg_el(i,j),j=1,nsp_el_erg)
           if (el_num.ne.lnods_num(i)) THEN
              call erro_init(myid,parallel,luerr)
              write(luerr,*)'FEHLER IM ERGEBNISFILE                '
              write(luerr,*)'Die Elementnumerierung im Ergebnis-File '
              write(luerr,*)'und im Geometrie-File ist unterschiedlich.'
              write(luerr,*)'Adresse                 :',i      
              write(luerr,*)'Nummer im Geometrie-File:',lnods_num(i)
              write(luerr,*)'Nummer im Ergebnis -File:',el_num      
              call erro_ende(myid,parallel,luerr)
           endif
 40   continue

      do 50 i=1,npoin_zal
           read(lu,*) (erg_za(i,j),j=1,nsp_za_erg)
 50   continue

      close(lu)
c     ****************************************************************



c     ****************************************************************
c     FEHLERMELDUNG FUER FALSCH GEOEFFNETE FILES:
      goto 888
 777  continue      
      comment='Fehler beim Oeffnen von File:'
      call erro_init(myid,parallel,luerr)
      write(luerr,*)'Fehler in Routine ERG_LES'
      call char_druck(comment,file_name,luerr)
      call erro_ende(myid,parallel,luerr)
 888  continue      
c     ****************************************************************


      return
      end 

