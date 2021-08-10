#ifndef REFERENCEFRAME_H
#define REFERENCEFRAME_H

#include "GlobalModel.h"
#include "Deformation.h"
#include "PoseMatch.h"
#include "Context.h"
#include "Ferns.h"
#include "Utils/Options.h"

#include <Eigen/Dense>

class ReferenceFrame
{
    public:
        ReferenceFrame()
        :m_localFerns(500, Options::get().depth * 1000, Options::get().interMapPhotoThresh),
         m_rgbd(Resolution::getInstance().width(),
                Resolution::getInstance().height(),
                Intrinsics::getInstance().cx(),
                Intrinsics::getInstance().cy(),
                Intrinsics::getInstance().fx(),
                Intrinsics::getInstance().fy()),
         m_firstRun(true)
        {
           
        }

        virtual ~ReferenceFrame()
        {
            //need to reclaim gpu memory here!!!
        }

        bool resolveRelativeTransformationFern(std::vector<Ferns::SurfaceConstraint, Eigen::aligned_allocator<Ferns::SurfaceConstraint>>& constraints,
                                               Eigen::Matrix4f & relativeTransform,
                                               Eigen::Matrix4f & currPose,
                                               GPUTexture & vertexTexture,
                                               GPUTexture & normalTexture,
                                               GPUTexture & imageTexture,
                                               const int & tick,
                                               const bool lost,
                                               const int depthCutoff,
                                               const float & confidenceThreshold,
                                               const int & timeIdx,
                                               const int & timeDelta,
                                               const int & maxTime)
        {
            Eigen::Matrix4f recoveryPose = m_localFerns.findFrame(constraints,
                                                                  currPose,
                                                                  &vertexTexture,
                                                                  &normalTexture,
                                                                  &imageTexture,
                                                                  tick,
                                                                  lost,
                                                                  depthCutoff,
                                                                  true);
            
            // Eigen::Matrix4f recoveryPose = m_localFerns.findFrameIntermap(
            //                                                       currPose,
            //                                                       &vertexTexture,
            //                                                       &normalTexture,
            //                                                       &imageTexture,
            //                                                       tick,
            //                                                       lost,
            //                                                       depthCutoff); 
        
            if(m_localFerns.lastClosest == -1)return false;
            std::cout << "found fern" << std::endl;
            //if(m_localFerns.lastClosestInterMap == -1)return false;            
            //relativeTransform = recoveryPose * currPose.inverse();

            m_index.combinedPredict(recoveryPose,
                                    m_localModel.model(),
                                    depthCutoff,
                                    confidenceThreshold,
                                    0,
                                    timeIdx,
                                    maxTime,
                                    timeDelta,
                                    IndexMap::INACTIVE);
            
            m_rgbd.initICPModel(m_index.oldVertexTex(), m_index.oldNormalTex(), (float)depthCutoff, recoveryPose);
            m_rgbd.initICP(&vertexTexture, &normalTexture, (float)depthCutoff);
            
            m_rgbd.initRGBModel(m_index.imageTex());
            m_rgbd.initRGB(&imageTexture);

            Eigen::Vector3f t = recoveryPose.topRightCorner(3, 1);
            Eigen::Matrix<float,3,3,Eigen::RowMajor> r = recoveryPose.topLeftCorner(3, 3);
            m_rgbd.getIncrementalTransformation(t, r, false, 10, true, false, true, true);
            
            recoveryPose.topRightCorner(3, 1) = t;
            recoveryPose.topLeftCorner(3, 3) = r;
            //currPose = recoveryPose;
            relativeTransform = recoveryPose * currPose.inverse();
            //m_poseMatches.push_back(PoseMatch(m_localFerns.lastClosest, m_localFerns.frames.size(), m_localFerns.frames.at(m_localFerns.lastClosest)->pose, recoveryPose, constraints, true));

            Eigen::MatrixXd covar = m_rgbd.getCovariance();
            bool covOk = true;

            for(int i = 0; i < 6; i++)
            {
                if(covar(i, i) > Options::get().covThresh)//
                {
                    covOk = false;
                    break;
                }
            }
            // std:: cout << "cov ok: " << covOk << " icp error" << m_rgbd.lastICPError << " icp count: " << m_rgbd.lastICPCount << std::endl;
            return  covOk && m_rgbd.lastICPError < Options::get().icpErrThresh && m_rgbd.lastICPCount > Options::get().icpCountThresh;//true;

            // Eigen::Matrix4f t = m_localFerns.findFrameIntermap(currPose,
            //                                                    &vertexTexture, &normalTexture, &imageTexture,
            //                                                    tick, lost, depthCutoff);

            // relativeTransform = t;

            // return m_localFerns.lastClosest != -1;
        }

        void consumeReferenceFrame(ReferenceFrame & other, Eigen::Matrix4f relativeTransform)
        {
            //m_localModel.consume(other.globalModel().cudaModel(), other.globalModel().lastCount(), relativeTransform);
            m_localModel.consume(other.globalModel().model(), relativeTransform);
            m_localFerns.consume(other.ferns().frames, relativeTransform, Options::get().fernThresh);//TODO use the same conservatory across all fern DBs
            
            Eigen::Matrix3f r = relativeTransform.topLeftCorner(3,3); 
            Eigen::Vector3f t = relativeTransform.topRightCorner(3,1);
            for(auto & kv : other.contexts())
            {
                kv.second->currPose() = relativeTransform * kv.second->currPose();
                m_contexts[kv.first] = kv.second;

                for(int i = 0; (size_t)i < kv.second->relativeCons().size(); i++)
                {
                    kv.second->relativeCons()[i].src = r * kv.second->relativeCons()[i].src + t;
                    kv.second->relativeCons()[i].target = r * kv.second->relativeCons()[i].target + t; 
                }

                for(int i = 0; (size_t)i < kv.second->poseGraph().size(); i++)
                {
                    kv.second->poseGraph()[i].second = relativeTransform * kv.second->poseGraph()[i].second;
                }

                for(int i = 0; (size_t)i < kv.second->miKeyframes().size(); i++)
                {
                    kv.second->miKeyframes()[i]->pose() = relativeTransform * kv.second->miKeyframes()[i]->pose();
                }
            }
        }

        // GPUTexture * getLastCrossMapPrediction()
        // {
        //     return m_index.vertexTex();
        // }

        GlobalModel & globalModel()        
        {
            return m_localModel;
        }
        const GlobalModel & globalModel() const
        {
            return m_localModel;
        }

        Deformation & globalDeformation()
        {
            return  m_globalDeformation;
        }
        const Deformation & globalDeformation() const
        {
            return  m_globalDeformation;
        }

        Deformation & localDeformation()
        {
            return m_localDeformation;    
        }

        const Deformation & localDeformation() const
        {
            return m_localDeformation;    
        }

        Ferns & ferns()
        {
            return m_localFerns;
        }
        const Ferns & ferns() const
        {
            return m_localFerns;
        }

        std::vector<PoseMatch, Eigen::aligned_allocator<PoseMatch> > & poseMatches()
        {
            return m_poseMatches;
        }

        bool & firstRun(){return m_firstRun;}
        std::map<std::string, std::shared_ptr<Context>> & contexts() {return m_contexts;}
        const std::map<std::string, std::shared_ptr<Context>> & contexts() const {return m_contexts;}
        std::string name;
    private:
        GlobalModel m_localModel;
        Deformation m_globalDeformation;
        Deformation m_localDeformation;
        Ferns m_localFerns;
        RGBDOdometry m_rgbd;
        IndexMap m_index;

        bool m_firstRun;

        std::vector<PoseMatch, Eigen::aligned_allocator<PoseMatch>> m_poseMatches;
        std::map<std::string, std::shared_ptr<Context>> m_contexts;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif /*REFERENCEFRAME_H*/