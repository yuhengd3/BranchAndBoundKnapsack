#ifndef __BranchAndBoundKnapsack_CUH__
#define __BranchAndBoundKnapsack_CUH__

#include "version.h"
#include "hostCode/AppDriverBase.cuh"

#include "io/Buffer.cuh"
#include "io/Sink.cuh"

#include "SubProblem.cuh"

class BranchAndBoundKnapsack {
public:
   static const int NUM_NODES = 9;
   
   // NODE IDENTIFIERS: 
   // 0 = A0node
   // 1 = A1node
   // 2 = A2node
   // 3 = A3node
   // 4 = A4node
   // 5 = A5node
   // 6 = A6node
   // 7 = A7node
   // 8 = SinkNode [SubProblem]
   // -1 = MERCATOR scheduler
   
   struct AppParams {
      unsigned * weights;
      unsigned * profits;
      double globalUpperBound;
      int maxCapacity;
      int maxItems;
   };
   
   AppParams* getParams()
   { return &allParams.appParams; }
   
   void setSource(const Mercator::Buffer<SubProblem> &buffer)
   { allParams.sourceBufferData = buffer.getData(); }
   
   class A {
   public:
      struct ModuleParams {
      };
      
      struct NodeParams {
      };
      
      ModuleParams* getParams()
      { return params; }
      
      A(ModuleParams *iparams)
       : params(iparams) {}
      
   private:
      ModuleParams * const params;
   };
   
   class __MTR_SINK_14410064 {
   public:
      struct ModuleParams {
      };
      
      struct NodeParams {
         Mercator::SinkData<SubProblem> sinkData;
      };
      
      ModuleParams* getParams()
      { return params; }
      
      __MTR_SINK_14410064(ModuleParams *iparams)
       : params(iparams) {}
      
   private:
      ModuleParams * const params;
   };
   
   class SinkNode {
   public:
      void setSink(Mercator::Buffer<SubProblem>& buffer)
      {
         params->sinkData.bufferData = buffer.getData();
      }
      
      SinkNode(__MTR_SINK_14410064::NodeParams *iparams)
       : params(iparams) {}
      
   private:
      __MTR_SINK_14410064::NodeParams * const params;
   };
   
   struct Params {
      Mercator::BufferData<SubProblem> *sourceBufferData;
      
      AppParams appParams;
      
      A::ModuleParams pA;
      A::NodeParams nA[8];
      __MTR_SINK_14410064::ModuleParams p__MTR_SINK_14410064;
      __MTR_SINK_14410064::NodeParams n__MTR_SINK_14410064[1];
   };
   
public:
   SinkNode SinkNode;
   
   BranchAndBoundKnapsack(cudaStream_t stream = 0, int deviceId = -1);
   
   int getNBlocks() const
   { return driver->getNBlocks(); }
   
   void run()
   { driver->run(&allParams); }
   
   void runAsync()
   { driver->runAsync(&allParams); }
   
   void join()
   { driver->join(); }
   
   ~BranchAndBoundKnapsack()
   {
      delete driver;
   }
   
private:
   Params allParams;
   
   Mercator::AppDriverBase<Params> *driver;
   
}; // end class BranchAndBoundKnapsack
#endif
