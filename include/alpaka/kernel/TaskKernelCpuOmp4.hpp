/* Copyright 2019 Benjamin Worpitz, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED

#if _OPENMP < 201307
    #error If ALPAKA_ACC_CPU_BT_OMP4_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#endif

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

// Implementation details.
#include <alpaka/acc/AccCpuOmp4.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/idx/MapIdx.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#include <alpaka/meta/ApplyTuple.hpp>

#include <omp.h>

#include <stdexcept>
#include <tuple>
#include <type_traits>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>
#endif

#include <typeinfo>
template<class T>
static void printArg(const T& t)
{
    std::cerr << "Arg: type=" << typeid(t).name() << '\n';
}

template<
    typename T>
static void printArg(T* t)
{
    std::cerr << "Ptr Arg: ptr=" << t << '\n';
// #pragma omp target enter data is_device_ptr(t)
}

static void printArgs() {}

template<class T, class... Args>
static void printArgs(const T& t, Args... args)
{
    printArg(t);
    printArgs(std::forward<Args>(args)...);
}

namespace alpaka
{
    namespace kernel
    {
        //#############################################################################
        //! The CPU OpenMP 4.0 accelerator execution task.
        template<
            typename TDim,
            typename TIdx,
            typename TKernelFnObj,
            typename... TArgs>
        class TaskKernelCpuOmp4 final :
            public workdiv::WorkDivMembers<TDim, TIdx>
        {
        public:
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST TaskKernelCpuOmp4(
                TWorkDiv && workDiv,
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args) :
                    workdiv::WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv)),
                    m_kernelFnObj(kernelFnObj),
                    m_args(args...)
            {
                static_assert(
                    dim::Dim<typename std::decay<TWorkDiv>::type>::value == TDim::value,
                    "The work division and the execution task have to be of the same dimensionality!");
                std::cout << "m_gridBlockExtent " << workDiv.m_gridBlockExtent << std::endl;
                std::cout << "m_blockThreadExtent " << workDiv.m_blockThreadExtent << std::endl;
                std::cout << "m_threadElemExtent " << workDiv.m_threadElemExtent << std::endl;
            }
            //-----------------------------------------------------------------------------
            TaskKernelCpuOmp4(TaskKernelCpuOmp4 const & other) = default;
            //-----------------------------------------------------------------------------
            TaskKernelCpuOmp4(TaskKernelCpuOmp4 && other) = default;
            //-----------------------------------------------------------------------------
            auto operator=(TaskKernelCpuOmp4 const &) -> TaskKernelCpuOmp4 & = default;
            //-----------------------------------------------------------------------------
            auto operator=(TaskKernelCpuOmp4 &&) -> TaskKernelCpuOmp4 & = default;
            //-----------------------------------------------------------------------------
            ~TaskKernelCpuOmp4() = default;

            //-----------------------------------------------------------------------------
            //! Executes the kernel function object.
            ALPAKA_FN_HOST auto operator()() const
            -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                auto const gridBlockExtent(
                    workdiv::getWorkDiv<Grid, Blocks>(*this));
                auto const blockThreadExtent(
                    workdiv::getWorkDiv<Block, Threads>(*this));
                auto const threadElemExtent(
                    workdiv::getWorkDiv<Thread, Elems>(*this));

                std::cout << "m_gridBlockExtent=" << this->m_gridBlockExtent << "\tgridBlockExtent=" << gridBlockExtent << std::endl;
                std::cout << "m_blockThreadExtent=" << this->m_blockThreadExtent << "\tblockThreadExtent=" << blockThreadExtent << std::endl;
                std::cout << "m_threadElemExtent=" << this->m_threadElemExtent << "\tthreadElemExtent=" << threadElemExtent << std::endl;

                // Get the size of the block shared dynamic memory.
                auto const blockSharedMemDynSizeBytes(
                    meta::apply(
                        [&](TArgs const & ... args)
                        {
                            return
                                kernel::getBlockSharedMemDynSizeBytes<
                                    acc::AccCpuOmp4<TDim, TIdx>>(
                                        m_kernelFnObj,
                                        blockThreadExtent,
                                        threadElemExtent,
                                        args...);
                        },
                        m_args));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__
                    << " blockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B" << std::endl;
#endif
                // The number of blocks in the grid.
                TIdx const gridBlockCount(gridBlockExtent.prod());
                // The number of threads in a block.
                TIdx const blockThreadCount(blockThreadExtent.prod());

                // We have to make sure, that the OpenMP runtime keeps enough threads for executing a block in parallel.
                auto const maxOmpThreadCount(::omp_get_max_threads());
                assert(blockThreadCount <= static_cast<TIdx>(maxOmpThreadCount));
                TIdx const teamCount(std::min(static_cast<TIdx>(maxOmpThreadCount)/blockThreadCount, gridBlockCount));
                // The number of elements in a thread. (to avoid mapping vec to
                // target, also fix range if maxTeamCount is limting)
                TIdx const threadElemCount(threadElemExtent[0u]);
                std::cout << "threadElemCount=" << threadElemCount << std::endl;
                std::cout << "teamCount=" << teamCount << "\tgridBlockCount=" << gridBlockCount << std::endl;

                if(::omp_in_parallel() != 0)
                {
                    throw std::runtime_error("The OpenMP 4.0 backend can not be used within an existing parallel region!");
                }

                // Force the environment to use the given number of threads.
                int const ompIsDynamic(::omp_get_dynamic());
                ::omp_set_dynamic(0);

                meta::apply([&](TArgs ... args){printArgs(std::forward<TArgs>(args)...);}, m_args);

                // `When an if(scalar-expression) evaluates to false, the structured block is executed on the host.`
                auto argsD = m_args;
                #pragma omp target map(to:argsD)
                {
                    #pragma omp teams num_teams(teamCount) thread_limit(blockThreadCount)
                    {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                        // The first team does some checks ...
                        if((::omp_get_team_num() == 0))
                        {
                            int const iNumTeams(::omp_get_num_teams());
                            printf("%s omp_get_num_teams: %d\n", __func__, iNumTeams);
                        }
#endif
                        printf("threadElemCount_dev %d\n", int(threadElemCount));
                        // iterate over groups of teams to stay withing thread limit
                        for(TIdx t = 0u; t < gridBlockCount; t+=teamCount)
                        {
                            acc::AccCpuOmp4<TDim, TIdx> acc(
                                threadElemCount,
                                gridBlockCount,
                                t,
                                blockSharedMemDynSizeBytes);

                            printf("acc->threadElemCount %d\n"
                                    , int(acc.m_threadElemExtent[0]));

                            const TIdx bsup = std::min(t + teamCount, gridBlockCount);
                            #pragma omp distribute
                            for(TIdx b = t; b<bsup; ++b)
                            {
                                vec::Vec<dim::DimInt<1u>, TIdx> const gridBlockIdx(b);
                                // When this is not repeated here:
                                // error: gridBlockExtent referenced in target region does not have a mappable type
                                auto const gridBlockExtent2(
                                    workdiv::getWorkDiv<Grid, Blocks>(*static_cast<workdiv::WorkDivMembers<TDim, TIdx> const *>(this)));
                                acc.m_gridBlockIdx = idx::mapIdx<TDim::value>(
                                    gridBlockIdx,
                                    gridBlockExtent2);

                                // Execute the threads in parallel.

                                // Parallel execution of the threads in a block is required because when syncBlockThreads is called all of them have to be done with their work up to this line.
                                // So we have to spawn one OS thread per thread in a block.
                                // 'omp for' is not useful because it is meant for cases where multiple iterations are executed by one thread but in our case a 1:1 mapping is required.
                                // Therefore we use 'omp parallel' with the specified number of threads in a block.
                                #pragma omp parallel num_threads(blockThreadCount)
                                {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                                    // The first thread does some checks in the first block executed.
                                    if((::omp_get_thread_num() == 0) && (b == 0))
                                    {
                                        int const numThreads(::omp_get_num_threads());
                                        printf("%s omp_get_num_threads: %d\n", __func__, numThreads);
                                        if(numThreads != static_cast<int>(blockThreadCount))
                                        {
                                            printf("ERROR: The OpenMP runtime did not use the number of threads that had been requested!\n");
                                        }
                                    }
#endif
                                    meta::apply(
                                        [&](TArgs ... args)
                                        {
                                            m_kernelFnObj(
                                                    acc,
                                                    args...);
                                        },
                                        argsD);

                                    // Wait for all threads to finish before deleting the shared memory.
                                    // This is done by default if the omp 'nowait' clause is missing
                                    //block::sync::syncBlockThreads(acc);
                                }

                                // After a block has been processed, the shared memory has to be deleted.
                                block::shared::st::freeMem(acc);
                            }
                        }
                    }
                }

                // Reset the dynamic thread number setting.
                ::omp_set_dynamic(ompIsDynamic);
            }

        private:
            TKernelFnObj m_kernelFnObj;
            std::tuple<TArgs...> m_args;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 execution task accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct AccType<
                kernel::TaskKernelCpuOmp4<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = acc::AccCpuOmp4<TDim, TIdx>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 execution task device type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevType<
                kernel::TaskKernelCpuOmp4<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = dev::DevCpu;
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 execution task dimension getter trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DimType<
                kernel::TaskKernelCpuOmp4<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = TDim;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 execution task platform type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct PltfType<
                kernel::TaskKernelCpuOmp4<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = pltf::PltfCpu;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 execution task idx type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct IdxType<
                kernel::TaskKernelCpuOmp4<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
