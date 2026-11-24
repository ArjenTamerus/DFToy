module parameters

  implicit none

  !TODO radius (cutoff_e) & grid size
  type :: dftoy_params
    integer :: num_wavevectors
    integer :: num_plane_waves
    integer :: num_states
    integer :: num_nl_states
    integer :: total_grid_size

    logical :: params_set = .false.
  end type

  type :: dftoy_options
    integer :: wavevectors
    integer :: states
    integer :: bands
    integer :: nl_states
    logical :: debug
    logical :: help
    logical :: usage
    logical :: version
  end type

  integer, parameter :: dp = selected_real_kind(15,300)
  integer, parameter :: cmplx_0 = cmplx(0.0_dp, 0.0_dp, dp)
  type(dftoy_params) :: params
  type(dftoy_options) :: options

  contains

    subroutine set_default_config()
      params%num_wavevectors = 2
      params%num_states = 2
      params%num_nl_states = 1

      params%num_plane_waves = 2 * params%num_wavevectors + 1
      params%total_grid_size = params%num_plane_waves*params%num_plane_waves*params%num_plane_waves
    end subroutine

    subroutine print_version()
      call dftoy_print("DFToy version TODO")
    end subroutine

    subroutine usage()
      call dftoy_print("TODO usages")
    end subroutine

    subroutine dftoy_print(msg)
      character(len=*) :: msg

      print *, msg
    end subroutine
end module
