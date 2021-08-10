#ifndef KEYFRAME_H_
#define KEYFRAME_H_

#include "GPUTexture.h"
#include "Cuda/types.cuh"

class KeyFrame
{
  public:
    KeyFrame(GPUTexture &img_tex,
             GPUTexture &vert_tex,
             GPUTexture &norm_tex,
             GPUTexture &depth_tex,
             const Eigen::Matrix4f &pose)
        : m_height(Resolution::getInstance().height()),
          m_width(Resolution::getInstance().width())
    //   m_img_tex(m_width, m_height, GL_RGBA, GL_RGB, GL_UNSIGNED_BYTE, true, true),
    //   m_vert_tex(m_width, m_height, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, true, true),
    //   m_norm_tex(m_width, m_height, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, true, true)
    {
        m_pose = pose;

        DeviceArray<float> m_vmap_tmp;
        //DeviceArray<float> m_nmap_tmp;
        DeviceArray<unsigned short> m_depth_tmp;

        m_vmap.create(m_height * 3, m_width);
        //m_nmap.create(m_height * 3, m_width);
        m_img.create(m_height, m_width);
        m_vmap_tmp.create(m_height * 4 * m_width);
        //m_nmap_tmp.create(m_height * 4 * m_width);
        m_dmap.create(m_height, m_width);
       
        cudaArray *textPtr;

        cudaGraphicsMapResources(1, &(vert_tex.cudaRes));
        cudaGraphicsSubResourceGetMappedArray(&textPtr, vert_tex.cudaRes, 0, 0);
        cudaMemcpyFromArray(m_vmap_tmp.ptr(), textPtr, 0, 0, m_vmap_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &(vert_tex.cudaRes));

        // cudaGraphicsMapResources(1, &(norm_tex.cudaRes));
        // cudaGraphicsSubResourceGetMappedArray(&textPtr, norm_tex.cudaRes, 0, 0);
        // cudaMemcpyFromArray(m_nmap_tmp.ptr(), textPtr, 0, 0, m_nmap_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
        // cudaGraphicsUnmapResources(1, &(norm_tex.cudaRes));

        //copyMaps(m_vmap_tmp, m_nmap_tmp, m_vmap, m_nmap);
        copyMaps(m_vmap_tmp, m_vmap);

        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcam = m_pose.topLeftCorner(3, 3);
        Eigen::Vector3f tcam = m_pose.topRightCorner(3, 1);

        mat33 device_Rcam = Rcam;
        float3 device_tcam = *reinterpret_cast<float3 *>(tcam.data());

        //tranformMaps(m_vmap, m_nmap, device_Rcam, device_tcam, m_vmap, m_nmap);
        tranformMaps(m_vmap, device_Rcam, device_tcam, m_vmap);

        cudaGraphicsMapResources(1, &(img_tex.cudaRes));
        cudaGraphicsSubResourceGetMappedArray(&textPtr, img_tex.cudaRes, 0, 0);
        imageBGRToIntensity(textPtr, m_img);
        cudaGraphicsUnmapResources(1, &img_tex.cudaRes);
        
        cudaDeviceSynchronize();

        cudaGraphicsMapResources(1, &(depth_tex.cudaRes));
        cudaGraphicsSubResourceGetMappedArray(&textPtr, depth_tex.cudaRes, 0, 0);
        cudaMemcpy2DFromArray(m_dmap.ptr(), m_dmap.step(), textPtr, 0, 0, m_dmap.colsBytes(), m_dmap.rows(), cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &depth_tex.cudaRes);

        m_vmap_tmp.release();
        //m_nmap_tmp.release();

        cudaDeviceSynchronize();

        // glCopyImageSubData(img_tex.texture->tid, GL_TEXTURE_BUFFER, GL_TEXTURE_2D_ARRAY, 0, 0, 0, m_img_tex.texture->tid, GL_TEXTURE_BUFFER, GL_TEXTURE_2D, 0, 0, 0,
        //                    m_width, m_height, 1);
        // glCopyImageSubData(vert_tex.texture->tid, GL_TEXTURE_BUFFER, GL_TEXTURE_2D_ARRAY, 0, 0, 0, m_vert_tex.texture->tid, GL_TEXTURE_BUFFER, GL_TEXTURE_2D, 0, 0, 0,
        //                    m_width, m_height * 3, 1);
        // glCopyImageSubData(norm_tex.texture->tid, GL_TEXTURE_BUFFER, GL_TEXTURE_2D_ARRAY, 0, 0, 0, m_norm_tex.texture->tid, GL_TEXTURE_BUFFER, GL_TEXTURE_2D, 0, 0, 0,
        //                    m_width, m_height * 3, 1);
    }

    KeyFrame( GPUTexture & activeImageTex,
              GPUTexture & activeVertexTex,
              GPUTexture & activeNormTex,
              GPUTexture & oldImageTex,
              GPUTexture & oldVertexTex,
              GPUTexture & oldNormTex,
              const Eigen::Matrix4f &pose,
              const float depthCutoff)
               :m_height(Resolution::getInstance().height()),
               m_width(Resolution::getInstance().width())
    {
        m_pose = pose;

        DeviceArray<float> m_vmap_tmp;
        DeviceArray<float> m_nmap_tmp;
        //DeviceArray<unsigned short> m_depth_tmp;

        m_vmap.create(m_height * 3, m_width);
        m_nmap_tmp.create(m_height * 4 * m_width);
        m_nmap.create(m_height * 3, m_width);
        m_img.create(m_height, m_width);
        m_vmap_tmp.create(m_height * 4 * m_width);
        
        m_dmap.create(m_height, m_width);
       
        cudaArray *textPtr;

        cudaGraphicsMapResources(1, &(activeVertexTex.cudaRes));
        cudaGraphicsSubResourceGetMappedArray(&textPtr, activeVertexTex.cudaRes, 0, 0);
        cudaMemcpyFromArray(m_vmap_tmp.ptr(), textPtr, 0, 0, m_vmap_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &(activeVertexTex.cudaRes));

        cudaGraphicsMapResources(1, &(activeNormTex.cudaRes));
        cudaGraphicsSubResourceGetMappedArray(&textPtr, activeNormTex.cudaRes, 0, 0);
        cudaMemcpyFromArray(m_nmap_tmp.ptr(), textPtr, 0, 0, m_nmap_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &activeNormTex.cudaRes);

        copyMaps(m_vmap_tmp, m_nmap_tmp, m_vmap, m_nmap);
        //copyMaps(m_vmap_tmp, m_vmap);

        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcam = m_pose.topLeftCorner(3, 3);
        Eigen::Vector3f tcam = m_pose.topRightCorner(3, 1);

        mat33 device_Rcam = Rcam;
        float3 device_tcam = *reinterpret_cast<float3 *>(tcam.data());

        tranformMaps(m_vmap, m_nmap, device_Rcam, device_tcam, m_vmap, m_nmap);
        //tranformMaps(m_vmap, device_Rcam, device_tcam, m_vmap);

        cudaGraphicsMapResources(1, &(activeImageTex.cudaRes));
        cudaGraphicsSubResourceGetMappedArray(&textPtr, activeImageTex.cudaRes, 0, 0);
        imageBGRToIntensity(textPtr, m_img);
        cudaGraphicsUnmapResources(1, &activeImageTex.cudaRes);
        
        cudaDeviceSynchronize();

        verticesToDepth(m_vmap_tmp, m_dmap, depthCutoff);

        m_vmap_tmp.release();
        m_nmap_tmp.release();

        DeviceArray<float> m_vmap_old_tmp;
        m_vmap_old_tmp.create(m_height * 4 * m_width);

        m_old_vmap.create(m_height * 3, m_width);
        m_old_img.create(m_height, m_width);
        m_old_dmap.create(m_height, m_width);

        cudaGraphicsMapResources(1, &(oldVertexTex.cudaRes));
        cudaGraphicsSubResourceGetMappedArray(&textPtr, oldVertexTex.cudaRes, 0, 0);
        cudaMemcpyFromArray(m_vmap_old_tmp.ptr(), textPtr, 0, 0, m_vmap_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &(oldVertexTex.cudaRes));

        copyMaps(m_vmap_old_tmp, m_old_vmap);

        tranformMaps(m_old_vmap, device_Rcam, device_tcam, m_old_vmap);

        cudaGraphicsMapResources(1, &(oldImageTex.cudaRes));
        cudaGraphicsSubResourceGetMappedArray(&textPtr, oldImageTex.cudaRes, 0, 0);
        imageBGRToIntensity(textPtr, m_old_img);
        cudaGraphicsUnmapResources(1, &oldImageTex.cudaRes);
        
        cudaDeviceSynchronize();

        verticesToDepth(m_vmap_old_tmp, m_old_dmap, depthCutoff);

        m_vmap_old_tmp.release();

        cudaDeviceSynchronize();
    }

    KeyFrame(GPUTexture & img_tex,
             GPUTexture & depth_tex,
             //GPUTexture &norm_tex,
             const Eigen::Matrix4f &pose,
             const int &depthCutoff,
             const float &fx, const float &fy,
             const float &cx, const float &cy)
        : m_height(Resolution::getInstance().height()),
          m_width(Resolution::getInstance().width())
    //   m_img_tex(m_width, m_height, GL_RGBA, GL_RGB, GL_UNSIGNED_BYTE, true, true),
    //   m_vert_tex(m_width, m_height, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, true, true),
    //   m_norm_tex(m_width, m_height, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, true, true)
    {
        m_pose = pose;

        DeviceArray<float> m_vmap_tmp;
        DeviceArray<float> m_nmap_tmp;
        DeviceArray2D<unsigned short> m_depth_tmp;

        m_vmap.create(m_height * 3, m_width);
        m_nmap.create(m_height * 3, m_width);
        m_img.create(m_height, m_width);

        m_vmap_tmp.create(m_height * 4 * m_width);
        m_nmap_tmp.create(m_height * 4 * m_width);
        m_depth_tmp.create(m_height, m_width);

        cudaArray *textPtr;

        cudaGraphicsMapResources(1, &(depth_tex.cudaRes));
        cudaGraphicsSubResourceGetMappedArray(&textPtr, depth_tex.cudaRes, 0, 0);
        cudaMemcpy2DFromArray(m_dmap.ptr(), m_dmap.step(), textPtr, 0, 0, m_dmap.colsBytes(), m_dmap.rows(), cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &(depth_tex.cudaRes));

        CameraModel intrinsics(fx, fy, cx, cy);
        createVMap(intrinsics, m_depth_tmp, m_vmap, depthCutoff);
        createNMap(m_vmap, m_nmap);

        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcam = m_pose.topLeftCorner(3, 3);
        Eigen::Vector3f tcam = m_pose.topRightCorner(3, 1);

        mat33 device_Rcam = Rcam;
        float3 device_tcam = *reinterpret_cast<float3 *>(tcam.data());

        tranformMaps(m_vmap, m_nmap, device_Rcam, device_tcam, m_vmap, m_nmap);
        // tranformMaps(m_vmap, device_Rcam, device_tcam, m_vmap);

        cudaGraphicsMapResources(1, &(img_tex.cudaRes));
        cudaGraphicsSubResourceGetMappedArray(&textPtr, img_tex.cudaRes, 0, 0);
        imageBGRToIntensity(textPtr, m_img);
        cudaGraphicsUnmapResources(1, &img_tex.cudaRes);

        m_vmap_tmp.release();
        m_nmap_tmp.release();
        m_depth_tmp.release();

        cudaDeviceSynchronize();
    }

    KeyFrame(const Eigen::Matrix4f &pose) : m_height(Resolution::getInstance().height()),
                                            m_width(Resolution::getInstance().width()) { m_pose = pose; }

    virtual ~KeyFrame()
    {
        m_vmap.release();
        m_nmap.release();
        m_img.release();
        m_dmap.release();
        m_old_vmap.release();
        //m_old_nmap.release();
        m_old_img.release();
        m_old_dmap.release();
    }

    
    void freemem()
    {
        m_vmap.release();
        m_nmap.release();
        m_img.release();
        m_dmap.release();
        m_old_vmap.release();
        m_old_nmap.release();
        m_old_img.release();
        m_old_dmap.release();
    }

    DeviceArray2D<unsigned char> &img() { return m_img; }
    DeviceArray2D<float> &vmap() { return m_vmap; }
    DeviceArray2D<float> &nmap() { return m_nmap; }
    DeviceArray2D<float> & dmap(){return m_dmap;}
    
    DeviceArray2D<unsigned char> &old_img() { return m_old_img; }
    DeviceArray2D<float> &old_vmap() { return m_old_vmap; }
    DeviceArray2D<float> &old_nmap() { return m_old_nmap; }
    DeviceArray2D<float> & old_dmap(){return m_old_dmap;}

    Eigen::Matrix4f pose() const { return m_pose; }

    Eigen::Matrix4f & pose() { return m_pose; }

  private:
    const int m_height;
    const int m_width;

    DeviceArray2D<float> m_vmap;
    DeviceArray2D<float> m_nmap;
    DeviceArray2D<unsigned char> m_img;
    DeviceArray2D<float> m_dmap;
    
    DeviceArray2D<float> m_old_vmap;
    DeviceArray2D<float> m_old_nmap;
    DeviceArray2D<unsigned char> m_old_img;
    DeviceArray2D<float> m_old_dmap;

    Eigen::Matrix4f m_pose;
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif /*KEYFRAME_H_*/