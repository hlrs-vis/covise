C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE MIT_DIM(mit_name,nsp_kn_erg,nsp_el_erg,nsp_za_erg)
 
      implicit none

      include 'common.zer'

      integer    i,lu,luerr,iread,lp,ii,
     *           nsp_kn_def,nsp_el_def,nsp_za_def,
     *           nsp_kn_erg,nsp_el_erg,nsp_za_erg,
     *           npoin_erg,nelem_erg,npoin_zal

      logical    default_kn_set,default_el_set,default_za_set,
     *           warn_print,erro_print,format_read

      character*80 file_name,mit_name,comment,reihe

      parameter (lu=66)
c     ****************************************************************


c     **********************************************************
c     INITIALISIERUNGEN:

      npoin_erg=0
      npoin_zal=0
      nelem_erg=0

      nsp_kn_erg=0
      nsp_el_erg=0
      nsp_za_erg=0

c     Setzen der Default-Werte:
      nsp_kn_def=6
      nsp_el_def=1
      nsp_za_def=5
c     **********************************************************


c     **********************************************************
      file_name=mit_name
      open(lu,file=mit_name,status='old',err=777)
      format_read=.true.
      einzelgeo=.false.
      rbenameread=.true.
      CALL HEAD_READ(lu,file_name,format_read,reihe)

      npoin_erg=iread(reihe)
      nelem_erg=iread(reihe)
      npoin_zal=iread(reihe)

      nsp_kn_erg=iread(reihe)
      nsp_el_erg=iread(reihe)
      nsp_za_erg=iread(reihe)

      close(lu)
c     **********************************************************

c     **********************************************************
c     FEHLERMELDUNGEN:

      erro_print=.false.        
      if (npoin.ne.npoin_erg.or.nelem.ne.nelem_erg) then
         erro_print=.true.
      endif
      if (npoin_zal.ne.0.and.npoin_zal.ne.npoin) then
         erro_print=.true.
      endif

      if (erro_print) then
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine MIT_DIM'
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
c     **********************************************************

c     ****************************************************************
c     SETZEN DER DEFAULT-WERTE WENN KEINE ANGABEN GEFUNDEN WURDEN:

      default_kn_set=.false.
      default_el_set=.false.
      default_za_set=.false.
      warn_print=.false.        
      if (nsp_kn_erg.eq.0.and.npoin_max.ne.0) then
         nsp_kn_erg=nsp_kn_def
         default_kn_set=.true.
         warn_print=.true.         
      endif
      if (nsp_el_erg.eq.0.and.nelem_max.ne.0) then
         nsp_el_erg=nsp_el_def
         default_el_set=.true.
         warn_print=.true.         
      endif
      if (nsp_za_erg.eq.0.and.npoin_zal.ne.0) then
         nsp_za_erg=nsp_za_def
         default_za_set=.true.
         warn_print=.true.         
      endif

      if (warn_print) then
        DO 345 ii=1,2
        if (ii.eq.1) then
           lp=6
        else if (ii.eq.2) then
           lp=lupro
        endif
        write(lp,*)'                                           '
        write(lp,*)'**************** W A R N U N G ****************'
        write(lp,*)'Bezueglich der Anzahl Ergebnis-Spalten wurden  ' 
        write(lp,*)'Default-Werte gesetzt.                         '

         if (default_kn_set) then
            write(lp,*)'Gesetzte    Anzahl  Knotenspalten:',nsp_kn_erg
         else
            write(lp,*)'Eingelesene Anzahl  Knotenspalten:',nsp_kn_erg
         endif

         if (default_el_set) then
            write(lp,*)'Gesetzte    Anzahl Elementspalten:',nsp_el_erg
         else 
            write(lp,*)'Eingelesene Anzahl Elementspalten:',nsp_el_erg
         endif

         if (default_za_set.and.npoin_zal.ne.0) then
            write(lp,*)'Gesetzte    Anzahl Altwertspalten:',nsp_za_erg
         else if (.not.default_za_set.and.npoin_zal.ne.0) then
            write(lp,*)'Eingelesene Anzahl Altwertspalten:',nsp_za_erg
         endif
         write(lp,*)'**************** W A R N U N G ****************'
         write(lp,*)'                                           '
 345     CONTINUE
      endif
c     ****************************************************************


c     **********************************************************
c     FEHLERMELDUNG FUER FALSCH GEOEFFNETE FILES:

      goto 888
 777  continue      
      comment='Fehler beim Oeffnen von File:'
      call erro_init(myid,parallel,luerr)
      call char_druck(comment,file_name,luerr)
      call erro_ende(myid,parallel,luerr)
 888  continue      
c     **********************************************************

      return
      end
