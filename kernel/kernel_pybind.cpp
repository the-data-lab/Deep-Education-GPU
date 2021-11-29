#include <pybind11/pybind11.h>

#include <pybind11/numpy.h>

#include <iostream>

using std::cout;
using std::endl;
#include "dlpack.h"

#include "kernel.h"

#include "op.h"

#include "invoke.h"

namespace py = pybind11;



array1d_t<float> capsule_to_array1d(py::capsule& capsule) 
{
    //DLManagedTensor * dlMTensor = (DLManagedTensor *)PyCapsule_GetPointer(&input, "dltesnor");
    //The worst static_cast, this is becacuse of 'operator T*()' in pybind11 code
    //Pybind11 should have simply provided GetPointer() API.

    DLManagedTensor * dlMTensor = static_cast<DLManagedTensor*>(capsule);
    assert(dlMTensor);
    DLTensor* tensor = &dlMTensor->dl_tensor;
    
    int64_t shape0 = tensor->shape[0];
    float*  data_ptr = (float*)tensor->data;

    array1d_t<float> array(data_ptr, shape0);
    return array;
}


array2d_t < float > capsule_to_array2d(py::capsule & capsule) {
    DLManagedTensor * dlMTensor = static_cast < DLManagedTensor * > (capsule);
    assert(dlMTensor);
    DLTensor * tensor = & dlMTensor -> dl_tensor;

    int64_t shape0 = tensor -> shape[0];
    int64_t shape1 = tensor -> shape[1];
    float * data_ptr = (float * ) tensor -> data;

    array2d_t < float > array(data_ptr, shape0, shape1);
    return array;

}

array3d_t<float> capsule_to_array3d(py::capsule& capsule) 
{
    DLManagedTensor * dlMTensor = static_cast<DLManagedTensor*>(capsule);
    assert(dlMTensor);
    DLTensor* tensor = &dlMTensor->dl_tensor;
    
    int64_t shape0 = tensor->shape[0];
    int64_t shape1 = tensor->shape[1];
    int64_t shape2 = tensor->shape[2];
    float*  data_ptr = (float*)tensor->data;

    array3d_t<float> array(data_ptr, shape0, shape1, shape2);
    return array;
}
PYBIND11_MODULE(kernel, m) {

    py::class_ < graph_t > (m, "graph_t")
        .def(py::init < > ())
        .def("get_vcount", & graph_t::get_vcount)
        .def("get_edge_count", &graph_t::get_edge_count)
        ;

        m.def("gspmm",
            [](graph_t & graph, py::capsule & input, py::capsule & output, bool reverse, bool norm) {
                array2d_t < float > input_array = capsule_to_array2d(input);
                array2d_t < float > output_array = capsule_to_array2d(output);
                int dim = input_array.col_count;
                if (reverse) {

                    return invoke_spmm(graph.csr, input_array, output_array, eSUM, reverse, norm, dim);
                } else {

                    return invoke_spmm(graph.csc, input_array, output_array, eSUM, reverse, norm, dim);
                }
            }
        );

    m.def("init_graph",
        [](py::array offset_csr, py::array nebrs_csr, py::array offset_csc, py::array nebrs_csc) {
            vid_t v_count = offset_csr.shape(0) - 1;
            vid_t dst_size = nebrs_csr.itemsize();
            vid_t * offset_csr1 = (vid_t * )(offset_csr.request().ptr);
            vid_t * offset_csc1 = (vid_t * ) offset_csc.request().ptr;
            void * nebrs_csr1 = nebrs_csr.request().ptr;
            void * nebrs_csc1 = nebrs_csc.request().ptr; 
            return invoke_init_graph(v_count, dst_size,
                offset_csr1, nebrs_csr1,
                offset_csc1, nebrs_csc1);
        }
    );
}
