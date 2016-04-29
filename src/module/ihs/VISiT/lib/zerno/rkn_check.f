C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE RKN_CHECK(kn_num,coord_zeig,typ_name)

      implicit none

      include 'common.zer'

      integer  kn_num,coord_zeig,luerr

      character*35 typ_name

      dimension coord_zeig(knmax_num)
c     ****************************************************************

c     ****************************************************************
      IF (kn_num.gt.knmax_num .or. coord_zeig(kn_num).eq.0) THEN
          call erro_init(myid,parallel,luerr)
          write(luerr,*)'FEHLER IN SUBROUTINE RKN_CHECK          '
          write(luerr,*)'Fehler bei ',typ_name                       
          write(luerr,*)'Die angegebene Knotennummer ', kn_num
          write(luerr,*)'existiert nicht im Geometrie-File.'
          write(luerr,*)'Max. Knotennummer im Geometrie-File:',knmax_num
          call erro_ende(myid,parallel,luerr)
      ENDIF
c     ****************************************************************

      return
      end

