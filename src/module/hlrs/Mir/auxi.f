!=====================================================================
!
!	This module provides auxiliary routines
!
!	Original Author: Michael M. Resch (MR)
!	First released: 30. December 1994
!
!=====================================================================
	module auxiliary
	contains

function parabel(y)
  use declaration
  use visco_declaration

  real(double)	:: y

  parabel = unull*2.0*(1-y**2)

end function parabel

function parabel_y(y)
  use declaration
  use visco_declaration

  real(double)	:: y

  parabel_y = unull*4.0*(-y)

end function parabel_y

function extrapolate(x_act,x_1,phi_1,x_2,phi_2,x_3,phi_3)

  use declaration

  implicit none

  real(double) :: x_act,x_1,x_2,x_3,phi_1,phi_2,phi_3
  real(double) :: a,b,c,det
  real(double) :: extrapolate

  det = x_1**2*x_2 +                                      &
        x_3**2*x_1 +                                      &
        x_2**2*x_3 -                                      &
        x_3**2*x_2 -                                      &
        x_2**2*x_1 -                                      &
        x_1**2*x_3

  if (det == 0.0) then
    write(*,*) 'singular matrix detected in drive z', x_act
    stop
  end if

  det = 1/det

  a = (x_2 - x_3)*phi_1 +                                 &
      (x_3 - x_1)*phi_2 +                                 &
      (x_1 - x_2)*phi_3

  a = a*det

  b = (x_3**2 - x_2**2)*phi_1 +                           &
      (x_1**2 - x_3**2)*phi_2 +                           &
      (x_2**2 - x_1**2)*phi_3

  b = b*det

  c = (x_2**2*x_3 - x_3**2*x_2)*phi_1 +                   &
      (x_3**2*x_1 - x_1**2*x_3)*phi_2 +                   &
      (x_1**2*x_2 - x_2**2*x_1)*phi_3

  c = c*det

  extrapolate = a*x_act**2 + b*x_act + c
        
end function extrapolate

	end module auxiliary
