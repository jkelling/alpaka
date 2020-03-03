/* Copyright 2019 Benjamin Worpitz, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED

#if _OPENACC < 201306
    #error If ALPAKA_ACC_ANY_BT_OACC_ENABLED is set, the compiler has to support OpenACC xx or higher!
#endif

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

// Implementation details.
#include <alpaka/acc/AccCpuOacc.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/idx/MapIdx.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#include <alpaka/meta/ApplyTuple.hpp>

#include <omp.h>

#include <functional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <algorithm>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>
#endif

namespace alpaka
{
    namespace kernel
    {
        namespace openacc
        {
            namespace detail
            {
                template<
                    typename TDim,
                    typename TIdx>
                struct AccCpuOpenACCWorker
                {
                    acc::AccCpuOacc<TDim, TIdx>& m_acc;
                    vec::Vec<TDim, TIdx> m_blockThreadIdx;

                        operator acc::AccCpuOacc<TDim, TIdx>& () {return m_acc;}
                        operator acc::AccCpuOacc<TDim, TIdx> const & () const {return m_acc;}

                        AccCpuOpenACCWorker(acc::AccCpuOacc<TDim, TIdx>& acc, const TIdx& wIdx) :
                            m_acc(acc),
                            m_blockThreadIdx(
                                idx::mapIdx<TDim::value>(
                                    vec::Vec<dim::DimInt<1u>, TIdx>(wIdx),
                                    workdiv::getWorkDiv<Block, Threads>(acc)
                                    )
                                )
                    {}
                };
            }
        }

        //#############################################################################
        //! The CPU OpenMP 4.0 accelerator execution task.
        template<
            typename TDim,
            typename TIdx,
            typename TKernelFnObj,
            typename... TArgs>
        class TaskKernelCpuOacc final :
            public workdiv::WorkDivMembers<TDim, TIdx>
        {
        public:
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST TaskKernelCpuOacc(
                TWorkDiv && workDiv,
                TKernelFnObj const & kernelFnObj,
                TArgs && ... args) :
                    workdiv::WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv)),
                    m_kernelFnObj(kernelFnObj),
                    m_args(std::forward<TArgs>(args)...)
            {
                static_assert(
                    dim::Dim<std::decay_t<TWorkDiv>>::value == TDim::value,
                    "The work division and the execution task have to be of the same dimensionality!");
            }
            //-----------------------------------------------------------------------------
            TaskKernelCpuOacc(TaskKernelCpuOacc const & other) = default;
            //-----------------------------------------------------------------------------
            TaskKernelCpuOacc(TaskKernelCpuOacc && other) = default;
            //-----------------------------------------------------------------------------
            auto operator=(TaskKernelCpuOacc const &) -> TaskKernelCpuOacc & = default;
            //-----------------------------------------------------------------------------
            auto operator=(TaskKernelCpuOacc &&) -> TaskKernelCpuOacc & = default;
            //-----------------------------------------------------------------------------
            ~TaskKernelCpuOacc() = default;

            //-----------------------------------------------------------------------------
            //! Executes the kernel function object.
            ALPAKA_FN_HOST auto operator()(
                    const
                    dev::DevOacc& dev
                ) const
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
                        [&](std::decay_t<TArgs> const & ... args)
                        {
                            return
                                kernel::getBlockSharedMemDynSizeBytes<
                                    acc::AccCpuOacc<TDim, TIdx>>(
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
                // We have to make sure, that the OpenMP runtime keeps enough threads for executing a block in parallel.
                TIdx const maxOmpThreadCount(static_cast<TIdx>(512));
                // The number of blocks in the grid.
                TIdx const gridBlockCount(gridBlockExtent.prod());
                // The number of threads in a block.
                TIdx const blockThreadCount(blockThreadExtent.prod());

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                if(maxOmpThreadCount < blockThreadExtent.prod())
                    std::cout << "Warning: TaskKernelCpuOacc: maxOmpThreadCount smaller than blockThreadCount requested by caller:" <<
                        maxOmpThreadCount << " < " << blockThreadExtent.prod() << std::endl;
#endif
                // make sure there is at least on team
                TIdx const teamCount(std::max(std::min(static_cast<TIdx>(maxOmpThreadCount/blockThreadCount), gridBlockCount), static_cast<TIdx>(1u)));
                std::cout << "threadElemCount=" << threadElemExtent[0u] << std::endl;
                std::cout << "teamCount=" << teamCount << "\tgridBlockCount=" << gridBlockCount << std::endl;

                // if(::omp_in_parallel() != 0)
                // {
                //     throw std::runtime_error("The OpenMP 4.0 backend can not be used within an existing parallel region!");
                // }

                // `When an if(scalar-expression) evaluates to false, the structured block is executed on the host.`
                auto argsD = m_args;
                auto kernelFnObj = m_kernelFnObj;
                dev.makeCurrent();
                #pragma acc parallel num_gangs(teamCount) num_workers(blockThreadCount)
                {
                    {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                        // The first team does some checks ...
                        if((::omp_get_team_num() == 0))
                        {
                            int const iNumTeams(::omp_get_num_teams());
                            printf("%s omp_get_num_teams: %d\n", __func__, iNumTeams);
                        }
#endif
                        printf("threadElemCount_dev %d\n", int(threadElemExtent[0u]));
                        // iterate over groups of teams to stay withing thread limit
                        for(TIdx t = 0u; t < gridBlockCount; t+=teamCount)
                        {
                            // printf("acc->threadElemCount %d\n"
                            //         , int(acc.m_threadElemExtent[0]));

                            const TIdx bsup = std::min(static_cast<TIdx>(t + teamCount), gridBlockCount);
                            #pragma acc loop gang
                            for(TIdx b = t; b<bsup; ++b)
                            {
                                vec::Vec<dim::DimInt<1u>, TIdx> const gridBlockIdx(b);
                                // When this is not repeated here:
                                // error: gridBlockExtent referenced in target region does not have a mappable type
                                auto const gridBlockExtent2(
                                    workdiv::getWorkDiv<Grid, Blocks>(*static_cast<workdiv::WorkDivMembers<TDim, TIdx> const *>(this)));
                                acc::AccCpuOacc<TDim, TIdx> acc(
                                    gridBlockExtent,
                                    blockThreadExtent,
                                    threadElemExtent,
                                    t,
                                    blockSharedMemDynSizeBytes);

                                acc.m_gridBlockIdx = idx::mapIdx<TDim::value>(
                                    gridBlockIdx,
                                    gridBlockExtent2);

                                // Execute the threads in parallel.

                                // Parallel execution of the threads in a block is required because when syncBlockThreads is called all of them have to be done with their work up to this line.
                                // So we have to spawn one OS thread per thread in a block.
                                // 'omp for' is not useful because it is meant for cases where multiple iterations are executed by one thread but in our case a 1:1 mapping is required.
                                // Therefore we use 'omp parallel' with the specified number of threads in a block.
                                // vec::Vec<dim::DimInt<1u>, TIdx> blockThreadIdx(static_cast<TIdx>(0u));
                                #pragma acc loop worker
                                for(TIdx w = 0; w < blockThreadCount; ++w)
                                {
                                    // blockThreadIdx[0] = w;
                                    auto wacc = typename openacc::detail::AccCpuOpenACCWorker<TDim, TIdx>(
                                            acc,
                                            w
                                            // idx::mapIdx<TDim::value>(
                                            //     blockThreadIdx,
                                            //     workdiv::getWorkDiv<Block, Threads>(
                                            //         *static_cast<workdiv::WorkDivMembers<TDim, TIdx> const *>(this)))
                                        );
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL && 0
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
                                        [kernelFnObj, &acc](typename std::decay<TArgs>::type const & ... args)
                                        {
                                            kernelFnObj(
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
            }

        private:
            TKernelFnObj m_kernelFnObj;
            std::tuple<std::decay_t<TArgs>...> m_args;
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
                kernel::TaskKernelCpuOacc<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = acc::AccCpuOacc<TDim, TIdx>;
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
                kernel::TaskKernelCpuOacc<TDim, TIdx, TKernelFnObj, TArgs...>>
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
                kernel::TaskKernelCpuOacc<TDim, TIdx, TKernelFnObj, TArgs...>>
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
                kernel::TaskKernelCpuOacc<TDim, TIdx, TKernelFnObj, TArgs...>>
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
                kernel::TaskKernelCpuOacc<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = TIdx;
            };
        }
    }

    namespace queue
    {
        namespace traits
        {
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct Enqueue<
                queue::QueueOaccBlocking,
                kernel::TaskKernelCpuOacc<TDim, TIdx, TKernelFnObj, TArgs...> >
            {
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueOaccBlocking& queue,
                    kernel::TaskKernelCpuOacc<TDim, TIdx, TKernelFnObj, TArgs...> const & task)
                -> void
                {
                    std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);

                    queue.m_spQueueImpl->m_bCurrentlyExecutingTask = true;

                    task(
                            queue.m_spQueueImpl->m_dev
                        );

                    queue.m_spQueueImpl->m_bCurrentlyExecutingTask = false;
                }
            };

            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct Enqueue<
                queue::QueueOaccNonBlocking,
                kernel::TaskKernelCpuOacc<TDim, TIdx, TKernelFnObj, TArgs...> >
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueOaccNonBlocking& queue,
                    kernel::TaskKernelCpuOacc<TDim, TIdx, TKernelFnObj, TArgs...> const & task)
                -> void
                {
                    queue.m_spQueueImpl->m_workerThread.enqueueTask(
                        [&queue, task]()
                        {
                            task(
                                    queue.m_spQueueImpl->m_dev
                                );
                        });
                }
            };
        }
    }
}

#endif
