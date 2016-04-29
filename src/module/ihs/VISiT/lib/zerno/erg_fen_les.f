C **************************************************************
c **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE ERG_FEN_LES(coord_num,lnods,
     *                       erg_kn,nsp_kn_erg,
     *                       erg_el,nsp_el_erg,
     *                       erg_name)
  
      implicit none     

      include 'common.zer'

      integer coord_num,nsp_kn_erg,nsp_el_erg,lnods

      integer i,j,k,lu,luerr,kn_num

      real    erg_kn,erg_el,werte(20),druck

      character*80 erg_name,comment,reihe,file_name

      parameter (lu=97)
C
      dimension coord_num(npoin_max),lnods(nelem_max,nkd),
     *          erg_kn(npoin_max,nsp_kn_erg),
     *          erg_el(nelem_max,nsp_el_erg)
c     *****************************************************************


c     ****************************************************************
c     EINLESEN DER ERGEBISSE:

      file_name=erg_name
      open(lu,file=file_name,status='old',err=777)


         do 21 i=1,10
            read(lu,'(A)') reihe
 21      continue

         druck=0.0
         do 31 i=1,npoin
           read(lu,*) kn_num,(erg_kn(i,j),j=1,nsp_kn_erg)

c    *                (werte(j),j=1,4),druck

           if (i.le.nelem) then
              erg_el(i,1)=druck
           endif

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
 31      continue

      close(lu)
c     ****************************************************************

c     ****************************************************************
c     BESTIMMUNG DES KNOTENDRUCKS:

      do 40 i=1,nelem
         druck=0.0
         do 41 k=1,nkd
            druck=druck+erg_kn(i,6)
 41      continue
         erg_el(i,1)=druck/REAL(nkd)
 40   continue
c     ****************************************************************



c     ****************************************************************
c     FEHLERMELDUNG FUER FALSCH GEOEFFNETE FILES:
      goto 888
 777  continue      
      comment='Fehler beim Oeffnen von File (erg_fen_les):'
      call erro_init(myid,parallel,luerr)
      write(luerr,*)'Fehler in Routine ERG_LES'
      call char_druck(comment,file_name,luerr)
      call erro_ende(myid,parallel,luerr)
 888  continue      
c     ****************************************************************


      return
      end 

