#ifndef __BranchAndBoundKnapsack_dev_CUH__
#define __BranchAndBoundKnapsack_dev_CUH__

#include "BranchAndBoundKnapsack.cuh"

#include "deviceCode/DeviceApp.cuh"
#include "deviceCode/NodeFunction_User.cuh"
#include "deviceCode/NodeFunction_Sink.cuh"
#include "deviceCode/Node_Source.cuh"
#include "deviceCode/Node_Queue.cuh"

#include "deviceCode/Scheduler_impl.cuh"
#include "io/SourceBase.cuh"

class BranchAndBoundKnapsack_dev : public Mercator::DeviceApp<9,128,8192,524288000> {
   class Source : public Mercator::SourceBase<SubProblem> {
      using BaseType = Mercator::SourceBase<SubProblem>;
      
      public:
      using EltT = BaseType::EltT;
      __device__
      Source(size_t *tailPtr, const BranchAndBoundKnapsack::Params* params)
       : BaseType(tailPtr),
         params(params)
      {}
      
      __device__
      void setup()
      {
         setStreamSize(params->sourceBufferData->size);
         data = params->sourceBufferData->data;
      }
      
      __device__
      EltT get(size_t idx) const { return data[idx]; }
      __device__
      void cleanup() {}
      
      private:
      const BranchAndBoundKnapsack::Params* const params;
      
      const SubProblem* data;
      
   };
   
   template <typename InputView>
   class A final : public Mercator::NodeFunction_User<SubProblem, 1, InputView, THREADS_PER_BLOCK, 1, 128, A> {
   public:
      // enum defining output channels
      struct Out {
         enum {
            __out = 0,
         };
      };
      
      __device__
      A(
        Mercator::RefCountedArena * parentArena,
        const BranchAndBoundKnapsack::AppParams* iappParams
       )
         : Mercator::NodeFunction_User<SubProblem, 1, InputView, THREADS_PER_BLOCK, 1, 128, A>(
                                                                                               parentArena
                                                                                              ),
         appParams(iappParams)
      {}
      
      __device__
      void run(SubProblem const & inputItem, unsigned int nInputs);
      
   private:
      
      using Mercator::NodeFunction_User<SubProblem, 1, InputView, THREADS_PER_BLOCK, 1, 128, A>::getNumActiveThreads;
      using Mercator::NodeFunction_User<SubProblem, 1, InputView, THREADS_PER_BLOCK, 1, 128, A>::getThreadGroupSize;
      using Mercator::NodeFunction_User<SubProblem, 1, InputView, THREADS_PER_BLOCK, 1, 128, A>::isThreadGroupLeader;
      using Mercator::NodeFunction_User<SubProblem, 1, InputView, THREADS_PER_BLOCK, 1, 128, A>::push;
      
      const BranchAndBoundKnapsack::AppParams* const appParams;
      
      __device__
      const BranchAndBoundKnapsack::AppParams* getAppParams() const
      { return appParams; }
      
      struct NodeState {
         double nodeUpperBound;
      };
      
      __device__
      NodeState* getState()
      { return &state; }
      
      NodeState state;
      
      public:
      __device__ void init(); // called once per block before run()
      __device__ void cleanup(); // called once per block after run()
   }; // end class A
   
   template <typename InputView>
   class __MTR_SINK_12722640 final : public Mercator::NodeFunction_Sink<SubProblem, InputView, THREADS_PER_BLOCK> {
   public:
      __device__
      __MTR_SINK_12722640(
                          Mercator::RefCountedArena * parentArena,
                          const BranchAndBoundKnapsack::__MTR_SINK_12722640::NodeParams* inodeParams,
                          const BranchAndBoundKnapsack::AppParams* iappParams
                         )
         : Mercator::NodeFunction_Sink<SubProblem, InputView, THREADS_PER_BLOCK>(
                                                                                 parentArena
                                                                                ),
         nodeParams(inodeParams),
         appParams(iappParams)
      {}
      
   private:
      
      const BranchAndBoundKnapsack::__MTR_SINK_12722640::NodeParams* const nodeParams;
      
      __device__
      const BranchAndBoundKnapsack::__MTR_SINK_12722640::NodeParams* getParams() const
      { return nodeParams; }
      
      const BranchAndBoundKnapsack::AppParams* const appParams;
      
      __device__
      const BranchAndBoundKnapsack::AppParams* getAppParams() const
      { return appParams; }
      
      public:
      __device__
      void init()
      {
         if (threadIdx.x == 0)
         {
            this->sink.init(nodeParams->sinkData.bufferData);
         }
      }
      
   }; // end class __MTR_SINK_12722640
   
public:
   __device__
   BranchAndBoundKnapsack_dev(size_t *tailPtr, const BranchAndBoundKnapsack::Params *params)
   {
      using Host = BranchAndBoundKnapsack;
      
      // construct each node of the app on the device
      A<Source>* dA0nodeFcn = new A<Source>(nullptr, &params->appParams);
      assert(dA0nodeFcn != nullptr);
      Source *sourceObj = new Source(tailPtr, params);
      Mercator::Node_Source<SubProblem, 1, Source, A> * dA0node = new Mercator::Node_Source<SubProblem, 1, Source, A>(&scheduler, 0, sourceObj, dA0nodeFcn);
      assert(dA0node != nullptr);
      
      A<Mercator::Queue<SubProblem>>* dA1nodeFcn = new A<Mercator::Queue<SubProblem>>(nullptr, &params->appParams);
      assert(dA1nodeFcn != nullptr);
      Mercator::Node_Queue<SubProblem, 1, 0, A> * dA1node = new Mercator::Node_Queue<SubProblem, 1, 0, A>(&scheduler, 0, 0, dA0node, 0, 1536, dA1nodeFcn);
      assert(dA1node != nullptr);
      
      A<Mercator::Queue<SubProblem>>* dA2nodeFcn = new A<Mercator::Queue<SubProblem>>(nullptr, &params->appParams);
      assert(dA2nodeFcn != nullptr);
      Mercator::Node_Queue<SubProblem, 1, 0, A> * dA2node = new Mercator::Node_Queue<SubProblem, 1, 0, A>(&scheduler, 0, 0, dA1node, 0, 1536, dA2nodeFcn);
      assert(dA2node != nullptr);
      
      A<Mercator::Queue<SubProblem>>* dA3nodeFcn = new A<Mercator::Queue<SubProblem>>(nullptr, &params->appParams);
      assert(dA3nodeFcn != nullptr);
      Mercator::Node_Queue<SubProblem, 1, 0, A> * dA3node = new Mercator::Node_Queue<SubProblem, 1, 0, A>(&scheduler, 0, 0, dA2node, 0, 1536, dA3nodeFcn);
      assert(dA3node != nullptr);
      
      A<Mercator::Queue<SubProblem>>* dA4nodeFcn = new A<Mercator::Queue<SubProblem>>(nullptr, &params->appParams);
      assert(dA4nodeFcn != nullptr);
      Mercator::Node_Queue<SubProblem, 1, 0, A> * dA4node = new Mercator::Node_Queue<SubProblem, 1, 0, A>(&scheduler, 0, 0, dA3node, 0, 1536, dA4nodeFcn);
      assert(dA4node != nullptr);
      
      A<Mercator::Queue<SubProblem>>* dA5nodeFcn = new A<Mercator::Queue<SubProblem>>(nullptr, &params->appParams);
      assert(dA5nodeFcn != nullptr);
      Mercator::Node_Queue<SubProblem, 1, 0, A> * dA5node = new Mercator::Node_Queue<SubProblem, 1, 0, A>(&scheduler, 0, 0, dA4node, 0, 1536, dA5nodeFcn);
      assert(dA5node != nullptr);
      
      A<Mercator::Queue<SubProblem>>* dA6nodeFcn = new A<Mercator::Queue<SubProblem>>(nullptr, &params->appParams);
      assert(dA6nodeFcn != nullptr);
      Mercator::Node_Queue<SubProblem, 1, 0, A> * dA6node = new Mercator::Node_Queue<SubProblem, 1, 0, A>(&scheduler, 0, 0, dA5node, 0, 1536, dA6nodeFcn);
      assert(dA6node != nullptr);
      
      A<Mercator::Queue<SubProblem>>* dA7nodeFcn = new A<Mercator::Queue<SubProblem>>(nullptr, &params->appParams);
      assert(dA7nodeFcn != nullptr);
      Mercator::Node_Queue<SubProblem, 1, 0, A> * dA7node = new Mercator::Node_Queue<SubProblem, 1, 0, A>(&scheduler, 0, 0, dA6node, 0, 1536, dA7nodeFcn);
      assert(dA7node != nullptr);
      
      __MTR_SINK_12722640<Mercator::Queue<SubProblem>>* dSinkNodeFcn = new __MTR_SINK_12722640<Mercator::Queue<SubProblem>>(nullptr, &params->n__MTR_SINK_12722640[0], &params->appParams);
      assert(dSinkNodeFcn != nullptr);
      Mercator::Node_Queue<SubProblem, 0, 0, __MTR_SINK_12722640> * dSinkNode = new Mercator::Node_Queue<SubProblem, 0, 0, __MTR_SINK_12722640>(&scheduler, 0, 1, dA7node, 0, 1536, dSinkNodeFcn);
      assert(dSinkNode != nullptr);
      
      // construct the output channels for each node
      {
         Mercator::Channel<SubProblem>*channel = new Mercator::Channel<SubProblem>(384, false, dA1node, dA1node->getQueue(), dA1node->getSignalQueue());
         assert(channel != nullptr);
         dA0node->setChannel(0, channel);
      }
      
      {
         Mercator::Channel<SubProblem>*channel = new Mercator::Channel<SubProblem>(384, false, dA2node, dA2node->getQueue(), dA2node->getSignalQueue());
         assert(channel != nullptr);
         dA1node->setChannel(0, channel);
      }
      
      {
         Mercator::Channel<SubProblem>*channel = new Mercator::Channel<SubProblem>(384, false, dA3node, dA3node->getQueue(), dA3node->getSignalQueue());
         assert(channel != nullptr);
         dA2node->setChannel(0, channel);
      }
      
      {
         Mercator::Channel<SubProblem>*channel = new Mercator::Channel<SubProblem>(384, false, dA4node, dA4node->getQueue(), dA4node->getSignalQueue());
         assert(channel != nullptr);
         dA3node->setChannel(0, channel);
      }
      
      {
         Mercator::Channel<SubProblem>*channel = new Mercator::Channel<SubProblem>(384, false, dA5node, dA5node->getQueue(), dA5node->getSignalQueue());
         assert(channel != nullptr);
         dA4node->setChannel(0, channel);
      }
      
      {
         Mercator::Channel<SubProblem>*channel = new Mercator::Channel<SubProblem>(384, false, dA6node, dA6node->getQueue(), dA6node->getSignalQueue());
         assert(channel != nullptr);
         dA5node->setChannel(0, channel);
      }
      
      {
         Mercator::Channel<SubProblem>*channel = new Mercator::Channel<SubProblem>(384, false, dA7node, dA7node->getQueue(), dA7node->getSignalQueue());
         assert(channel != nullptr);
         dA6node->setChannel(0, channel);
      }
      
      {
         Mercator::Channel<SubProblem>*channel = new Mercator::Channel<SubProblem>(384, false, dSinkNode, dSinkNode->getQueue(), dSinkNode->getSignalQueue());
         assert(channel != nullptr);
         dA7node->setChannel(0, channel);
      }
      
      
      Mercator::NodeBase *nodes[] = {dA0node, dA1node, dA2node, dA3node, dA4node, dA5node, dA6node, dA7node, dSinkNode, };
      // tell device app about all nodes
      registerNodes(nodes);
   }
}; // end class BranchAndBoundKnapsack_dev

#define __MDECL__ \
   template <typename InputView> __device__ 

#endif
