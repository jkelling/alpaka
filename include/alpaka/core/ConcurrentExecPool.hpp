/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

//-----------------------------------------------------------------------------
// Clang does not support exceptions when natively compiling device code.
// This is no problem at some places but others explicitly rely on std::exception_ptr,
// std::current_exception, std::make_exception_ptr, etc. which are not declared in device code.
// Therefore, we can not even parse those parts when compiling device code.
//-----------------------------------------------------------------------------
#include <alpaka/core/Common.hpp>   // BOOST_LANG_CUDA, BOOST_ARCH_CUDA_DEVICE

#include <boost/predef.h>           // workarounds
#include <boost/config.hpp>         // BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION

#include <queue>                    // std::queue
#include <mutex>                    // std::mutex
#include <stdexcept>                // std::current_exception
#include <vector>                   // std::vector
#include <exception>                // std::runtime_error
#include <utility>                  // std::forward
#include <atomic>                   // std::atomic
#include <future>                   // std::future

namespace alpaka
{
    namespace core
    {
        namespace detail
        {
            //#############################################################################
            //!
            //#############################################################################
            template<
                typename T>
            class ThreadSafeQueue :
                private std::queue<T>
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ThreadSafeQueue(
                    std::size_t)
                {}
                //-----------------------------------------------------------------------------
                //! \return If the queue is empty.
                //-----------------------------------------------------------------------------
                auto empty() const
                -> bool
                {
                    return std::queue<T>::empty();
                }
                //-----------------------------------------------------------------------------
                //! Pushes the given value onto the back of the queue.
                //-----------------------------------------------------------------------------
                auto push(
                    T && t)
                -> void
                {
                    std::lock_guard<std::mutex> lk(m_Mutex);

                    std::queue<T>::push(std::forward<T>(t));
                }
                //-----------------------------------------------------------------------------
                //! Pops the given value from the front of the queue.
                //-----------------------------------------------------------------------------
                auto pop(
                    T & t)
                -> bool
                {
                    std::lock_guard<std::mutex> lk(m_Mutex);

                    if(std::queue<T>::empty())
                    {
                        return false;
                    }
                    else
                    {
                        t = std::queue<T>::front();
                        std::queue<T>::pop();
                        return true;
                    }
                }

            private:
                std::mutex m_Mutex;
            };

            //#############################################################################
            //! ITaskPkg.
            //#############################################################################
            // \NOTE: We can not use C++11 std::packaged_task as it forces the use of std::future
            // but we additionally support boost::fibers::promise.
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wweak-vtables"
#endif
            class ITaskPkg
            {
            public:
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                virtual ~ITaskPkg() = default;

                //-----------------------------------------------------------------------------
                //! Runs this task.
                //-----------------------------------------------------------------------------
                auto runTask() noexcept
                -> void
                {
                    try
                    {
                        run();
                    }
                    catch(...)
                    {
// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_CUDA_DEVICE)
                        setException(std::current_exception());
#endif
                    }
                }

            private:
                //-----------------------------------------------------------------------------
                //! The execution function.
                //-----------------------------------------------------------------------------
                virtual auto run() -> void = 0;

            public:
// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_CUDA_DEVICE)
                //-----------------------------------------------------------------------------
                //! Sets an exception.
                //-----------------------------------------------------------------------------
                virtual auto setException(
                    std::exception_ptr const & exceptPtr)
                -> void = 0;
#endif
            };
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

            //#############################################################################
            //! TaskPkg with return type.
            //!
            //! \tparam TPromise The promise type returned by the task.
            //! \tparam TFnObj The type of the function to execute.
            //! \tparam TFnObjReturn The return type of the TFnObj. Used for class specialization.
            //#############################################################################
            template<
                template<typename TFnObjReturn> class TPromise,
                typename TFnObj,
                typename TFnObjReturn>
            class TaskPkg :
                public ITaskPkg
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                TaskPkg(
                    TFnObj && func) :
                        m_Promise(),
                        m_FnObj(std::forward<TFnObj>(func))
                {}

            private:
                //-----------------------------------------------------------------------------
                //! The execution function.
                //-----------------------------------------------------------------------------
                virtual auto run()
                -> void final
                {
                    m_Promise.set_value(this->m_FnObj());
                }
            public:
// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_CUDA_DEVICE)
                //-----------------------------------------------------------------------------
                //! Sets an exception.
                //-----------------------------------------------------------------------------
                virtual auto setException(
                    std::exception_ptr const & exceptPtr)
                -> void final
                {
                    m_Promise.set_exception(exceptPtr);
                }
#endif
                TPromise<TFnObjReturn> m_Promise;
            private:
                // NOTE: To avoid invalid memory accesses to memory of a different thread
                // `std::remove_reference` enforces the function object to be copied.
                typename std::remove_reference<TFnObj>::type m_FnObj;
            };

            //#############################################################################
            //! TaskPkg without return type.
            //!
            //! \tparam TPromise The promise type returned by the task.
            //! \tparam TFnObj The type of the function to execute.
            //#############################################################################
            template<
                template<typename TFnObjReturn> class TPromise,
                typename TFnObj>
            class TaskPkg<
                TPromise,
                TFnObj,
                void> :
                public ITaskPkg
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                TaskPkg(
                    TFnObj && func) :
                        m_Promise(),
                        m_FnObj(std::forward<TFnObj>(func))
                {}

            private:
                //-----------------------------------------------------------------------------
                //! The execution function.
                //-----------------------------------------------------------------------------
                virtual auto run()
                -> void final
                {
                    this->m_FnObj();
                    m_Promise.set_value();
                }
            public:
// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_CUDA_DEVICE)
                //-----------------------------------------------------------------------------
                //! Sets an exception.
                //-----------------------------------------------------------------------------
                virtual auto setException(
                    std::exception_ptr const & exceptPtr)
                -> void final
                {
                    m_Promise.set_exception(exceptPtr);
                }
#endif
                TPromise<void> m_Promise;
            private:
                // NOTE: To avoid invalid memory accesses to memory of a different thread
                // `std::remove_reference` enforces the function object to be copied.
                typename std::remove_reference<TFnObj>::type m_FnObj;
            };

            //#############################################################################
            //! ConcurrentExecPool using yield.
            //!
            //! \tparam TConcurrentExec The type of concurrent executor (for example std::thread).
            //! \tparam TPromise The promise type returned by the task.
            //! \tparam TYield The type is required to have a static method "void yield()" to yield the current thread if there is no work.
            //! \tparam TMutex Unused. The mutex type used for locking threads.
            //! \tparam TCondVar Unused. The condition variable type used to make the threads wait if there is no work.
            //! \tparam TisYielding Boolean value if the threads should yield instead of wait for a condition variable.
            //#############################################################################
            template<
                typename TSize,
                typename TConcurrentExec,
                template<typename TFnObjReturn> class TPromise,
                typename TYield,
                typename TMutex = void,
                typename TCondVar = void,
                bool TisYielding = true>
            class ConcurrentExecPool final
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //!
                //! Creates a concurrent executor pool with a specific number of concurrent executors and a maximum number of queued tasks.
                //!
                //! \param concurrentExecutionCount
                //!    The guaranteed number of concurrent executors used in the pool.
                //!    This is also the maximum number of tasks worked on concurrently.
                //! \param queueSize
                //!    The maximum number of tasks that can be queued for completion.
                //!    Currently running tasks do not belong to the queue anymore.
                //-----------------------------------------------------------------------------
                ConcurrentExecPool(
                    TSize concurrentExecutionCount,
                    TSize queueSize = 128u) :
                    m_vConcurrentExecs(),
                    m_qTasks(static_cast<std::size_t>(queueSize)),
                    m_bShutdownFlag(false)
                {
                    if(concurrentExecutionCount < 1)
                    {
                        throw std::invalid_argument("The argument 'concurrentExecutionCount' has to be greate or equal to one!");
                    }
                    if(queueSize < 1)
                    {
                        throw std::invalid_argument("The argument 'queueSize' has to be greate or equal to one!");
                    }

                    m_vConcurrentExecs.reserve(static_cast<std::size_t>(concurrentExecutionCount));

                    // Create all concurrent executors.
                    for(TSize concurrentExec(0u); concurrentExec < concurrentExecutionCount; ++concurrentExec)
                    {
                        m_vConcurrentExecs.emplace_back(std::bind(&ConcurrentExecPool::concurrentExecFn, this));
                    }
                }
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ConcurrentExecPool(ConcurrentExecPool const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ConcurrentExecPool(ConcurrentExecPool &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                auto operator=(ConcurrentExecPool const &) -> ConcurrentExecPool & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                auto operator=(ConcurrentExecPool &&) -> ConcurrentExecPool & = delete;

                //-----------------------------------------------------------------------------
                //! Destructor
                //!
                //! Completes any currently running task as normal.
                //! Signals a std::runtime_error exception to any other tasks that were not able to run.
                //-----------------------------------------------------------------------------
                ~ConcurrentExecPool()
                {
                    // Signal that concurrent executors should not perform any new work
                    m_bShutdownFlag.store(true);

                    joinAllConcurrentExecs();

                    auto currentTaskPackage(std::shared_ptr<ITaskPkg>{nullptr});

                    // Signal to each incomplete task that it will not complete due to pool destruction.
                    while(popTask(currentTaskPackage))
                    {
                        auto const except(std::runtime_error("Could not perform task before ConcurrentExecPool destruction"));
// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_CUDA_DEVICE)
                        currentTaskPackage->setException(std::make_exception_ptr(except));
#endif
                    }
                }

                //-----------------------------------------------------------------------------
                //! Runs the given function on one of the pool in First In First Out (FIFO) order.
                //!
                //! \tparam TFnObj   The function type.
                //! \param task     Function object to be called on the pool.
                //!                 Takes an arbitrary number of arguments and arbitrary return type.
                //! \tparam TArgs   The argument types pack.
                //! \param args     Arguments for task, cannot be moved.
                //!                 If such parameters must be used, use a lambda and capture via move then move the lambda.
                //!
                //! \return Signals when the task has completed with either success or an exception.
                //!         Also results in an exception if the pool is destroyed before execution has begun.
                //-----------------------------------------------------------------------------
                template<
                    typename TFnObj,
                    typename ... TArgs>
                auto enqueueTask(
                    TFnObj && task,
                    TArgs && ... args)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(std::declval<TPromise<decltype(task(args...))>>().get_future())
#endif
                {
                    auto boundTask(std::bind(task, args...));

                    // Return type of the function object, can be void via specialization of TaskPkg.
                    using FnObjReturn = decltype(task(args...));
                    using TaskPackage = TaskPkg<TPromise, decltype(boundTask), FnObjReturn>;

                    auto pTaskPackage(new TaskPackage(std::move(boundTask)));
                    std::shared_ptr<ITaskPkg> upTaskPackage(pTaskPackage);

                    auto future(pTaskPackage->m_Promise.get_future());

                    m_qTasks.push(std::move(upTaskPackage));

                    return future;
                }
                //-----------------------------------------------------------------------------
                //! \return The number of concurrent executors available.
                //-----------------------------------------------------------------------------
                auto getConcurrentExecutionCount() const
                -> TSize
                {
                    return m_vConcurrentExecs.size();
                }
                //-----------------------------------------------------------------------------
                //! \return If the work queue is empty.
                //-----------------------------------------------------------------------------
                auto isQueueEmpty() const
                -> bool
                {
                    return m_qTasks.empty();
                }

            private:
                //-----------------------------------------------------------------------------
                //! The function the concurrent executors are executing.
                //-----------------------------------------------------------------------------
                void concurrentExecFn()
                {
                    // Checks whether pool is being destroyed, if so, stop running.
                    while(!m_bShutdownFlag.load(std::memory_order_relaxed))
                    {
                        auto currentTaskPackage(std::shared_ptr<ITaskPkg>{nullptr});

                        // Use popTask so we only ever have one reference to the ITaskPkg
                        if(popTask(currentTaskPackage))
                        {
                            currentTaskPackage->runTask();
                        }
                        else
                        {
                            TYield::yield();
                        }
                    }
                }

                //-----------------------------------------------------------------------------
                //! Joins all concurrent executors.
                //-----------------------------------------------------------------------------
                void joinAllConcurrentExecs()
                {
                    for(auto && concurrentExec : m_vConcurrentExecs)
                    {
                        concurrentExec.join();
                    }
                }
                //-----------------------------------------------------------------------------
                //! Pops a task from the queue.
                //-----------------------------------------------------------------------------
                auto popTask(
                    std::shared_ptr<ITaskPkg> & out)
                -> bool
                {
                    if(m_qTasks.pop(out))
                    {
                        return true;
                    }
                    return false;
                }

            private:
                std::vector<TConcurrentExec> m_vConcurrentExecs;
                ThreadSafeQueue<std::shared_ptr<ITaskPkg>> m_qTasks;
                std::atomic<bool> m_bShutdownFlag;
            };

            //#############################################################################
            //! ConcurrentExecPool using a condition variable to wait for new work.
            //!
            //! \tparam TConcurrentExec The type of concurrent executor (for example std::thread).
            //! \tparam TPromise The promise type returned by the task.
            //! \tparam TYield Unused. The type is required to have a static method "void yield()" to yield the current thread if there is no work.
            //! \tparam TMutex The mutex type used for locking threads.
            //! \tparam TCondVar The condition variable type used to make the threads wait if there is no work.
            //#############################################################################
            template<
                typename TSize,
                typename TConcurrentExec,
                template<typename TFnObjReturn> class TPromise,
                typename TYield,
                typename TMutex,
                typename TCondVar>
            class ConcurrentExecPool<
                TSize,
                TConcurrentExec,
                TPromise,
                TYield,
                TMutex,
                TCondVar,
                false> final
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //!
                //! Creates a concurrent executor pool with a specific number of concurrent executors and a maximum number of queued tasks.
                //!
                //! \param concurrentExecutionCount
                //!    The guaranteed number of concurrent executors used in the pool.
                //!    This is also the maximum number of tasks worked on concurrently.
                //! \param queueSize
                //!    The maximum number of tasks that can be queued for completion.
                //!    Currently running tasks do not belong to the queue anymore.
                //-----------------------------------------------------------------------------
                ConcurrentExecPool(
                    TSize concurrentExecutionCount,
                    TSize queueSize = 128u) :
                    m_vConcurrentExecs(),
                    m_qTasks(static_cast<std::size_t>(queueSize)),
                    m_mtxWakeup(),
                    m_cvWakeup(),
                    m_bShutdownFlag(false)
                {
                    if(concurrentExecutionCount < 1)
                    {
                        throw std::invalid_argument("The argument 'concurrentExecutionCount' has to be greate or equal to one!");
                    }
                    if(queueSize < 1)
                    {
                        throw std::invalid_argument("The argument 'queueSize' has to be greate or equal to one!");
                    }

                    m_vConcurrentExecs.reserve(static_cast<std::size_t>(concurrentExecutionCount));

                    // Create all concurrent executors.
                    for(TSize concurrentExec(0u); concurrentExec < concurrentExecutionCount; ++concurrentExec)
                    {
                        m_vConcurrentExecs.emplace_back(std::bind(&ConcurrentExecPool::concurrentExecFn, this));
                    }
                }
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ConcurrentExecPool(ConcurrentExecPool const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ConcurrentExecPool(ConcurrentExecPool &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                auto operator=(ConcurrentExecPool const &) -> ConcurrentExecPool & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                auto operator=(ConcurrentExecPool &&) -> ConcurrentExecPool & = delete;

                //-----------------------------------------------------------------------------
                //! Destructor
                //!
                //! Completes any currently running task as normal.
                //! Signals a std::runtime_error exception to any other tasks that were not able to run.
                //-----------------------------------------------------------------------------
                ~ConcurrentExecPool()
                {
                    {
                        std::unique_lock<TMutex> lock(m_mtxWakeup);

                        // Signal that concurrent executors should not perform any new work
                        m_bShutdownFlag = true;
                    }

                    m_cvWakeup.notify_all();

                    joinAllConcurrentExecs();

                    auto currentTaskPackage(std::shared_ptr<ITaskPkg>{nullptr});

                    // Signal to each incomplete task that it will not complete due to pool destruction.
                    while(popTask(currentTaskPackage))
                    {
                        auto const except(std::runtime_error("Could not perform task before ConcurrentExecPool destruction"));
// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_CUDA_DEVICE)
                        currentTaskPackage->setException(std::make_exception_ptr(except));
#endif
                    }
                }

                //-----------------------------------------------------------------------------
                //! Runs the given function on one of the pool in First In First Out (FIFO) order.
                //!
                //! \tparam TFnObj   The function type.
                //! \param task     Function object to be called on the pool.
                //!                 Takes an arbitrary number of arguments and arbitrary return type.
                //! \tparam TArgs   The argument types pack.
                //! \param args     Arguments for task, cannot be moved.
                //!                 If such parameters must be used, use a lambda and capture via move then move the lambda.
                //!
                //! \return Signals when the task has completed with either success or an exception.
                //!         Also results in an exception if the pool is destroyed before execution has begun.
                //-----------------------------------------------------------------------------
                template<
                    typename TFnObj,
                    typename ... TArgs>
                auto enqueueTask(
                    TFnObj && task,
                    TArgs && ... args)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(std::declval<TPromise<decltype(task(args...))>>().get_future())
#endif
                {
                    auto boundTask(std::bind(task, args...));

                    // Return type of the function object, can be void via specialization of TaskPkg.
                    using FnObjReturn = decltype(task(args...));
                    using TaskPackage = TaskPkg<TPromise, decltype(boundTask), FnObjReturn>;

                    auto pTaskPackage(new TaskPackage(std::move(boundTask)));
                    std::shared_ptr<ITaskPkg> upTaskPackage(pTaskPackage);

                    auto future(pTaskPackage->m_Promise.get_future());

                    {
                        std::lock_guard<TMutex> lock(m_mtxWakeup);
                        m_qTasks.push(std::move(upTaskPackage));

                        m_cvWakeup.notify_one();
                    }

                    return future;
                }
                //-----------------------------------------------------------------------------
                //! \return The number of concurrent executors available.
                //-----------------------------------------------------------------------------
                auto getConcurrentExecutionCount() const
                -> TSize
                {
                    return m_vConcurrentExecs.size();
                }
                //-----------------------------------------------------------------------------
                //! \return If the work queue is empty.
                //-----------------------------------------------------------------------------
                auto isQueueEmpty() const
                -> bool
                {
                    return m_qTasks.empty();
                }

            private:
                //-----------------------------------------------------------------------------
                //! The function the concurrent executors are executing.
                //-----------------------------------------------------------------------------
                void concurrentExecFn()
                {
                    // Checks whether pool is being destroyed, if so, stop running (lazy check without mutex).
                    while(!m_bShutdownFlag)
                    {
                        auto currentTaskPackage(std::shared_ptr<ITaskPkg>{nullptr});

                        // Use popTask so we only ever have one reference to the ITaskPkg
                        if(popTask(currentTaskPackage))
                        {
                            currentTaskPackage->runTask();
                        }
                        {
                            std::unique_lock<TMutex> lock(m_mtxWakeup);
                            if(m_qTasks.empty())
                            {
                                // If the shutdown flag has been set since the last check, return now.
                                if(m_bShutdownFlag)
                                {
                                    return;
                                }

                                m_cvWakeup.wait(lock, [this]() { return ((!m_qTasks.empty()) || m_bShutdownFlag); });
                            }
                        }
                    }
                }

                //-----------------------------------------------------------------------------
                //! Joins all concurrent executors.
                //-----------------------------------------------------------------------------
                void joinAllConcurrentExecs()
                {
                    for(auto && concurrentExec : m_vConcurrentExecs)
                    {
                        concurrentExec.join();
                    }
                }
                //-----------------------------------------------------------------------------
                //! Pops a task from the queue.
                //-----------------------------------------------------------------------------
                auto popTask(
                    std::shared_ptr<ITaskPkg> & out)
                -> bool
                {
                    if(m_qTasks.pop(out))
                    {
                        return true;
                    }
                    return false;
                }

            private:
                std::vector<TConcurrentExec> m_vConcurrentExecs;
                ThreadSafeQueue<std::shared_ptr<ITaskPkg>> m_qTasks;

                TMutex m_mtxWakeup;
                TCondVar m_cvWakeup;
                std::atomic<bool> m_bShutdownFlag;
            };
        }
    }
}
