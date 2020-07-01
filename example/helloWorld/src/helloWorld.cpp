/* Copyright 2019 Benjamin Worpitz, Erik Zenker
 *
 * This file exemplifies usage of Alpaka.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED “AS IS” AND ISC DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <iostream>
#include <memory>
#include <utility>
#include <string>
#include <map>

#define PMACC_CASSERT_MSG(...)
#define HDINLINE inline
#define float_X double
#include "pmacc/math/ConstVector.hpp"

#define CONST_VECTOR(type,dim,name,...) PMACC_CONST_VECTOR(type,dim,name,__VA_ARGS__)

// CONST_VECTOR( float_X, 3, DriftParamElectrons_direction, 0.0, 0.0, 1.0 );
CONST_VECTOR( float_X, 3, DriftParamIons_direction, 0.0, 0.0, -1.0 );

//#############################################################################
//! Hello World Kernel
//!
//! Prints "[x, y, z][gtid] Hello World" where tid is the global thread number.
struct HelloWorldKernel
{
    //-----------------------------------------------------------------------------
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        int i) const
    -> void
    {
        constexpr ConstArrayStorage<double, 3> stor;
        printf(
            "%f %f\n"
            , DriftParamIons_direction_data[2]
            , DriftParamIons_direction_data[2]
            , stor[2]
            // , picongpu::particles::manipulators::DriftParamIons_direction.x()
            );
    }
};

#if 0
struct ITask
{
    virtual ~ITask() {}

    virtual void exec() = 0;
};

template<class Acc, class Queue, class WorkDiv>
class TaskHello : public ITask
{
    Queue* m_queue;
    WorkDiv m_workDiv;
    HelloWorldKernel m_helloWorldKernel;
    int m_i;

    public:

    TaskHello(Queue* queue, const WorkDiv& workDiv, int i = 1)
        : m_queue(queue), m_workDiv(workDiv), m_i(i)
    {}
    
    void exec() override
    {
        alpaka::kernel::exec<Acc>(
            *m_queue,
            m_workDiv,
            m_helloWorldKernel,
            m_i
            /* put kernel arguments here */);
        alpaka::wait::wait(*m_queue);
    }
};

struct TaskNop : public ITask
{
    void exec() override {std::cout << "nop" << std::endl;}
};

// template<class Acc, class Queue, class WorkDiv>
// ITask* taskFactory(int argc, char* argv[], Queue* queue, const WorkDiv& workDiv)
// {
//     if(argc > 1)
//         return new TaskHello<Acc, Queue, WorkDiv>(queue, workDiv);
//     else
//         return new TaskNop();
// }
// 
struct ITaskMap
{
    std::map<std::string, std::unique_ptr<ITask>> m_map;
    virtual ~ITaskMap() {}

    virtual void exec(const std::string& a) = 0;
    virtual void create(const std::string& a) = 0;
};

template<class Acc, class Queue, class WorkDiv>
struct TaskMap : public ITaskMap
{
    Queue* m_queue;
    WorkDiv m_workDiv;

    TaskMap(Queue* queue, const WorkDiv& workDiv)
        : m_queue(queue), m_workDiv(workDiv)
    {}

    void exec(const std::string& a) override
    {
        auto& task = m_map[a];
        if(task)
            task->exec();
    }

    void create(const std::string& a) override
    {
        if(a == "hello")
        {
            m_map["hello"].reset(new TaskHello<Acc, Queue, WorkDiv>(m_queue, m_workDiv));
        }
        else
        {
            m_map[a] = std::make_unique<TaskNop>();
        }
    }
};
#endif

auto main(int argc, char* argv[])
-> int
{
// Fallback for the CI with disabled sequential backend
#if defined(ALPAKA_CI) && !defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    return EXIT_SUCCESS;
#else
    // Define the index domain
    //
    // Depending on your type of problem, you have to define
    // the dimensionality as well as the type used for indices.
    // For small index domains 16 or 32 bit indices may be enough
    // and may be faster to calculate depending on the accelerator.
    using Dim = alpaka::dim::DimInt<3>;
    using Idx = std::size_t;

    // Define the accelerator
    //
    // It is possible to choose from a set of accelerators
    // that are defined in the alpaka::acc namespace e.g.:
    // - AccGpuCudaRt
    // - AccGpuHipRt
    // - AccCpuThreads
    // - AccCpuFibers
    // - AccCpuOmp2Threads
    // - AccCpuOmp2Blocks
    // - AccOmp5
    // - AccCpuTbbBlocks
    // - AccCpuSerial
    //
    // Each accelerator has strengths and weaknesses. Therefore,
    // they need to be choosen carefully depending on the actual
    // use case. Furthermore, some accelerators only support a
    // particular workdiv, but workdiv can also be generated
    // automatically.

    // By exchanging the Acc and Queue types you can select where to execute the kernel.
    // using Acc = alpaka::acc::AccCpuSerial<Dim, Idx>;
    using Acc = alpaka::example::ExampleDefaultAcc<Dim, Idx>;
    std::cout << "Using alpaka accelerator: " << alpaka::acc::getAccName<Acc>() << std::endl;

    // Defines the synchronization behavior of a queue
    //
    // choose between Blocking and NonBlocking
    using QueueProperty = alpaka::queue::Blocking;
    using Queue = alpaka::queue::Queue<Acc, QueueProperty>;

    // Select a device
    //
    // The accelerator only defines how something should be
    // parallized, but a device is the real entity which will
    // run the parallel programm. The device can be choosen
    // by id (0 to the number of devices minus 1) or you
    // can also retrieve all devices in a vector (getDevs()).
    // In this example the first devices is choosen.
    auto const devAcc = alpaka::pltf::getDevByIdx<Acc>(0u);

    // Create a queue on the device
    //
    // A queue can be interpreted as the work queue
    // of a particular device. Queues are filled with
    // tasks and alpaka takes care that these
    // tasks will be executed. Queues are provided in
    // non-blocking and blocking variants.
    // The example queue is a blocking queue to a cpu device,
    // but it also exists an non-blocking queue for this
    // device (QueueCpuNonBlocking).
    Queue queue(devAcc);

    // Define the work division
    //
    // A kernel is executed for each element of a
    // n-dimensional grid distinguished by the element indices.
    // The work division defines the number of kernel instantiations as
    // well as the type of parallelism used by the kernel execution task.
    // Different accelerators have different requirements on the work
    // division. For example, the sequential accelerator can not
    // provide any thread level parallelism (synchronizable as well as non synchronizable),
    // whereas the CUDA accelerator can spawn hundreds of synchronizing
    // and non synchronizing threads at the same time.
    //
    // The workdiv is divided in three levels of parallelization:
    // - grid-blocks:      The number of blocks in the grid (parallel, not synchronizable)
    // - block-threads:    The number of threads per block (parallel, synchronizable).
    //                     Each thread executes one kernel invocation.
    // - thread-elements:  The number of elements per thread (sequential, not synchronizable).
    //                     Each kernel has to execute its elements sequentially.
    //
    // - Grid     : consists of blocks
    // - Block    : consists of threads
    // - Elements : consists of elements
    //
    // Threads in the same grid can access the same global memory,
    // while threads in the same block can access the same shared
    // memory. Elements are supposed to be used for vectorization.
    // Thus, a thread can process data element size wise with its
    // vector processing unit.
    using Vec = alpaka::vec::Vec<Dim, Idx>;
    Vec const elementsPerThread(Vec::all(static_cast<Idx>(1)));
    Vec const threadsPerBlock(Vec::all(static_cast<Idx>(1)));
    Vec const blocksPerGrid(
        static_cast<Idx>(4),
        static_cast<Idx>(2),
        static_cast<Idx>(2));

    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>;
    WorkDiv const workDiv(
        blocksPerGrid,
        threadsPerBlock,
        elementsPerThread);


#if 1
    // Instantiate the kernel function object
    //
    // Kernels can be everything that has a callable operator()
    // and which takes the accelerator as first argument.
    // So a kernel can be a class or struct, a lambda, a std::function, etc.
    HelloWorldKernel helloWorldKernel;

    // Run the kernel
    //
    // To execute the kernel, you have to provide the
    // work division as well as the additional kernel function
    // parameters.
    // The kernel execution task is enqueued into an accelerator queue.
    // The queue can be blocking or non-blocking
    // depending on the choosen queue type (see type definitions above).
    // Here it is synchronous which means that the kernel is directly executed.
    alpaka::kernel::exec<Acc>(
        queue,
        workDiv,
        helloWorldKernel,argc
        /* put kernel arguments here */);
    alpaka::wait::wait(queue);
#else
    auto map = std::unique_ptr<ITaskMap>(new TaskMap<Acc, Queue, WorkDiv>(&queue, workDiv));
    map->create("nop");
    map->create("hello");
    if(argc > 1)
        map->exec(argv[1]);
    // auto task = taskFactory<Acc, Queue, WorkDiv>(argc, argv, &queue, workDiv);
    // task->exec();
    // delete task;
#endif

    return EXIT_SUCCESS;
#endif
}
