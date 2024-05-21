/*
 * Parallel.c
 *
 * Support routines facilitating parallel compute/communication.
 *
 */

//#include <stdbool.h>
//#include <stdlib.h>
//#include <stdio.h>
//#include <stdarg.h>
//#include <complex.h>
//#include "sycl_parallel.h"
//#include "trace.h"

#include <complex>
#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"
#include "interfaces.h"
#include "sycl_interfaces.h"

void sycl_fft_transpose(sycl::buffer<std::complex<double>> in,
		sycl::buffer<std::complex<double>> out,
		ptrdiff_t local_start, ptrdiff_t local_z, ptrdiff_t num_plane_waves,
		int direction)
{
	//size_t x, y, z;
	//size_t nb, count;
	//int rank;
	//int xdim_nblocks = world_size;
	//ptrdiff_t xdim_block = local_z;
	//ptrdiff_t ydim = num_plane_waves;


	sycl::queue q(sycl::default_selector_v);

	sycl::event e = q.submit([&](sycl::handler &h) {
			sycl::accessor in_acc(in, h, sycl::read_only);
			sycl::accessor out_acc(out, h, sycl::write_only);

			h.parallel_for(sycl::range<3>(num_plane_waves,num_plane_waves,num_plane_waves),
					[=](sycl::item<3> xyz) {
					size_t x = xyz.get_id(0);
					size_t y = xyz.get_id(1);
					size_t z = xyz.get_id(2);

					out_acc[x*num_plane_waves*num_plane_waves+y*num_plane_waves+z] =
						in_acc[xyz.get_linear_id()];

					});
			});

	e.wait();

	// NOTE: These are not _necessary_ - can reuse in and out if they're allocated
	// to the appropriate size. To keep the code more readable though, this should
	// only be done if profiling demonstrates an *actual* speed-up or memory
	// savings.
	//
	// Very rough estimate from watching _htop_ says - maybe 5% memory savings at
	// -s 24 -w 24? No noticable speed diff.
	// Would have to test for larger inputs, and run some profiles.
	//std::complex<double> *send, *recv;

	//if (direction == XZ) {
	//	send = calloc(local_z*ydim*(xdim_nblocks*xdim_block), sizeof(fftw_complex));
	//	recv = calloc(local_z*ydim*(xdim_nblocks*xdim_block), sizeof(fftw_complex));

	//	count = 0;
	//	for(nb = 0; nb < xdim_nblocks; nb++) {
	//		for(z = 0; z < local_z; z++) {
	//			for(y = 0; y < ydim; y++) {
	//				for(x = 0; x < xdim_block; x++) {
	//					if((nb*xdim_block+x >= num_plane_waves)) {
	//						send[count] = 0;
	//						count++;
	//					}
	//					else {
	//						send[count] = in[nb*xdim_block+z*ydim*ydim
	//							+ y*ydim + x];
	//						count++;
	//					}
	//				}
	//			}
	//		}
	//	}

	//	MPI_Alltoall(send, local_z*ydim*xdim_block*2, MPI_DOUBLE,
	//			recv, local_z*ydim*xdim_block*2, MPI_DOUBLE, MPI_COMM_WORLD);

	//	count = 0;
	//	for(nb = 0; nb < xdim_nblocks; nb++) {
	//		for(z = 0; z < local_z; z++) {
	//			for(y = 0; y < ydim; y++) {
	//				for(x = 0; x < xdim_block; x++) {
	//					if((nb*xdim_block+z >= num_plane_waves)) {
	//						count++;
	//					}
	//					else {
	//						 out[nb*xdim_block+x*ydim*ydim+y*ydim+z] = recv[count];
	//						 count++;
	//					}
	//				}
	//			}
	//		}
	//	}

	//	free(send);
	//	free(recv);
	//}
	//else {
	//	mpi_error("Invalid transpose direction.\n!");
	//}
}

