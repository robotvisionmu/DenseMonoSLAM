/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 *
 * The use of the code within this file and all code within files that
 * make up the software that is ElasticFusion is permitted for
 * non-commercial purposes only.  The full terms and conditions that
 * apply to the code within this file are detailed within the LICENSE.txt
 * file and at
 * <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/>
 * unless explicitly stated.  By downloading this file you agree to
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#ifndef GUI_H_
#define GUI_H_

#include <GPUTexture.h>
#include <Shaders/Shaders.h>
#include <Utils/Intrinsics.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/pangolin.h>
#include <deque>
#include <map>

#include "LogReader.h"

#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049

class GUI {
 public:
  GUI(bool liveCap, bool showcaseMode, std::vector<std::string> cams)
      : showcaseMode(showcaseMode) {
    width = 1280;
    height = 980;
    panel = 205;

    width += panel;

    pangolin::Params windowParams;

    windowParams.Set("SAMPLE_BUFFERS", 0);
    windowParams.Set("SAMPLES", 0);

    pangolin::CreateWindowAndBind("Main", width, height, windowParams);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    // Internally render at 3840x2160
    renderBuffer = new pangolin::GlRenderBuffer(3840, 2160),
    colorTexture = new GPUTexture(renderBuffer->width, renderBuffer->height,
                                  GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, true);

    colorFrameBuffer = new pangolin::GlFramebuffer;
    colorFrameBuffer->AttachColour(*colorTexture->texture);
    colorFrameBuffer->AttachDepth(*renderBuffer);

    colorProgram = std::shared_ptr<Shader>(loadProgramFromFile(
        "draw_global_surface.vert", "draw_global_surface_phong.frag",
        "draw_global_surface.geom"));
    fxaaProgram = std::shared_ptr<Shader>(
        loadProgramFromFile("empty.vert", "fxaa.frag", "quad.geom"));

    pangolin::SetFullscreen(showcaseMode);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);

    pangolin::Display(GPUTexture::RGB).SetAspect(1024.0f/320.0f);

    pangolin::Display(GPUTexture::DEPTH_NORM).SetAspect(1024.0f/320.0f);

    pangolin::Display("ModelImage").SetAspect(1024.0f/320.0f);

    pangolin::Display("Model").SetAspect(1024.0f/320.0f);

    std::vector<std::string> mi_labels;
    mi_labels.push_back(std::string("nid-rgb"));
    mi_labels.push_back(std::string("nid-depth"));
    mi_labels.push_back(std::string("threshold"));
    miLog.SetLabels(mi_labels);

    miPlot = new pangolin::Plotter(&miLog, 0, 600, -1, 2., 30, 0.5);
    miPlot->Track("$i");

    std::vector<std::string> labels;
    labels.push_back(std::string("residual"));
    labels.push_back(std::string("threshold"));
    resLog.SetLabels(labels);

    resPlot = new pangolin::Plotter(&resLog, 0, 300, 0, 0.0005, 30, 0.5);
    resPlot->Track("$i");

    std::vector<std::string> labels2;
    labels2.push_back(std::string("inliers"));
    labels2.push_back(std::string("threshold"));
    inLog.SetLabels(labels2);

    inPlot = new pangolin::Plotter(&inLog, 0, 300, 0, 40000, 30, 0.5);
    inPlot->Track("$i");

    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(panel));

    pangolin::Display("multi")
        .SetBounds(pangolin::Attach::Pix(0), 3 / 4.0f, 4 / 5.0f, 1.0)
        .SetLayout(pangolin::LayoutEqualVertical);
    for (auto& c : cams) {
      pangolin::Display(c).SetAspect(1024.0f/320.0f);
      pangolin::Display("multi")
          .AddDisplay(pangolin::Display(c))
          .AddDisplay(pangolin::Display(GPUTexture::RGB))
          .AddDisplay(pangolin::Display(GPUTexture::DEPTH_NORM))
          .AddDisplay(pangolin::Display("ModelImage"))
          .AddDisplay(pangolin::Display("Model"));
    }

    pangolin::Display("Map").SetBounds(
        0.0, 1.0, pangolin::Attach::Pix(panel), 1.0,
        -1024.0/320.0);

    pause = new pangolin::Var<bool>("ui.Pause", false, true);
    step = new pangolin::Var<bool>("ui.Step", false, false);
    save = new pangolin::Var<bool>("ui.Save", false, false);
    save_images = new pangolin::Var<bool>("ui.Save Images", false, false);
    reset = new pangolin::Var<bool>("ui.Reset", false, false);
    batchAlign = new pangolin::Var<bool>("ui.Batch Align", false, false);
    flipColors = new pangolin::Var<bool>("ui.Flip RGB", false, true);

    if (liveCap) {
      autoSettings = new pangolin::Var<bool>("ui.Auto Settings", true, true);
    } else {
      autoSettings = 0;
    }

    pyramid = new pangolin::Var<bool>("ui.Pyramid", true, true);
    so3 = new pangolin::Var<bool>("ui.SO(3)", true, true);
    frameToFrameRGB =
        new pangolin::Var<bool>("ui.Frame to frame RGB", false, true);
    fastOdom = new pangolin::Var<bool>("ui.Fast Odometry", false, true);
    rgbOnly = new pangolin::Var<bool>("ui.RGB only tracking", false, true);
    confidenceThreshold =
        new pangolin::Var<float>("ui.Confidence threshold", 10.0, 0.0, 24.0);
    depthCutoff = new pangolin::Var<float>("ui.Depth cutoff", 3.0, 0.0, 12.0);
    icpWeight = new pangolin::Var<float>("ui.ICP weight", 10.0, 0.0, 100.0);
    nidThreshold = new pangolin::Var<float>("ui.NID Threshold", 0.5, 0.0, 1.0);
    nidDepthWeight = new pangolin::Var<float>("ui.NID Depth Weight", 0.5, 0.0, 1.0);
    numBinsImg = new pangolin::Var<int>("ui.Num Bins Img", 64, 1, 256);
    numBinsDepth = new pangolin::Var<int>("ui.Num Bins Depth", 500, 1, 5000);
    nidPyramidLevel = new pangolin::Var<int>("ui.NID pyramid level", 0, 0, 2);
    followPose = new pangolin::Var<bool>("ui.Follow pose", true, true);
    drawRawCloud = new pangolin::Var<bool>("ui.Draw raw", false, true);
    drawFilteredCloud =
        new pangolin::Var<bool>("ui.Draw filtered", false, true);
    drawGlobalModel =
        new pangolin::Var<bool>("ui.Draw global model", true, true);
    drawUnstable =
        new pangolin::Var<bool>("ui.Draw unstable points", false, true);
    drawPoints = new pangolin::Var<bool>("ui.Draw points", false, true);
    drawColors = new pangolin::Var<bool>("ui.Draw colors", showcaseMode, true);
    drawFxaa = new pangolin::Var<bool>("ui.Draw FXAA", showcaseMode, true);
    drawWindow = new pangolin::Var<bool>("ui.Draw time window", false, true);
    drawNormals = new pangolin::Var<bool>("ui.Draw normals", false, true);
    drawTimes = new pangolin::Var<bool>("ui.Draw times", false, true);
    drawDefGraph =
        new pangolin::Var<bool>("ui.Draw deformation graph", false, true);
    drawFerns = new pangolin::Var<bool>("ui.Draw ferns", false, true);
    drawDeforms = new pangolin::Var<bool>("ui.Draw deformations", true, true);
    drawContributions =
        new pangolin::Var<bool>("ui.Draw contributions", false, true);

    gpuMem = new pangolin::Var<int>("ui.GPU memory free", 0);
    numKfs = new pangolin::Var<int>("ui.Number of KFs", 0);
    totalPoints = new pangolin::Var<std::string>("ui.Total points", "0");
    totalNodes = new pangolin::Var<std::string>("ui.Total nodes", "0");
    totalFerns = new pangolin::Var<std::string>("ui.Total ferns", "0");
    totalDefs = new pangolin::Var<std::string>("ui.Total deforms", "0");
    totalFernDefs =
        new pangolin::Var<std::string>("ui.Total fern deforms", "0");

    trackInliers = new pangolin::Var<std::string>("ui.Inliers", "0");
    trackRes = new pangolin::Var<std::string>("ui.Residual", "0");
    logProgress = new pangolin::Var<std::string>("ui.Log", "0");

    if (showcaseMode) {
      pangolin::RegisterKeyPressCallback(
          ' ', pangolin::SetVarFunctor<bool>("ui.Reset", true));
    }
  }

  virtual ~GUI() {
    delete pause;
    delete reset;
    delete inPlot;
    delete resPlot;

    if (autoSettings) {
      delete autoSettings;
    }
    delete step;
    delete save;
    delete save_images;
    delete trackInliers;
    delete trackRes;
    delete confidenceThreshold;
    delete totalNodes;
    delete drawWindow;
    delete so3;
    delete totalFerns;
    delete totalDefs;
    delete depthCutoff;
    delete logProgress;
    delete drawTimes;
    delete drawFxaa;
    delete fastOdom;
    delete icpWeight;
    delete pyramid;
    delete rgbOnly;
    delete totalFernDefs;
    delete drawFerns;
    delete followPose;
    delete drawDeforms;
    delete drawRawCloud;
    delete totalPoints;
    delete frameToFrameRGB;
    delete flipColors;
    delete drawFilteredCloud;
    delete drawNormals;
    delete drawColors;
    delete drawGlobalModel;
    delete drawUnstable;
    delete drawPoints;
    delete drawDefGraph;
    delete gpuMem;
    delete drawContributions;

    delete renderBuffer;
    delete colorFrameBuffer;
    delete colorTexture;

    delete batchAlign;
    delete nidThreshold;
    delete nidDepthWeight;
    delete numBinsImg;
    delete numBinsDepth;
    delete nidPyramidLevel;

    for (auto c : m_cameras) {
      delete c.second;
    }
  }

  void createViews(std::vector<std::shared_ptr<ReferenceFrame>> rfs) {
    s_cams.clear();
    ctxIdToDisplay.clear();
    pangolin::Display("Map").views.clear();

    std::deque<std::tuple<float, float, float, float>> bounds;
    bounds.push_back(std::make_tuple(0.0f, 1.0f, 0.0f, 1.0f));
    bool cutCol = false;  // true;
    while (bounds.size() < (size_t)rfs.size()) {
      std::tuple<float, float, float, float>& bound = bounds.front();
      if (cutCol) {
        bounds.push_back(std::make_tuple(std::get<0>(bound), std::get<1>(bound),
                                         std::get<2>(bound),
                                         std::get<3>(bound) / 2.0f));
        bounds.push_back(std::make_tuple(std::get<0>(bound), std::get<1>(bound),
                                         std::get<3>(bound) / 2.0f,
                                         std::get<3>(bound)));
      } else {
        bounds.push_back(
            std::make_tuple(std::get<0>(bound), std::get<1>(bound) / 2.0f,
                            std::get<2>(bound), std::get<3>(bound)));
        bounds.push_back(std::make_tuple(std::get<1>(bound) / 2.0f,
                                         std::get<1>(bound), std::get<2>(bound),
                                         std::get<3>(bound)));
      }
      bounds.pop_front();

      cutCol = ((int)bounds.size() && ((int)bounds.size() - 1)) == 0;
    }

    for (int i = 0; (size_t)i < rfs.size(); i++) {
      std::tuple<float, float, float, float> bound = bounds[i];
      pangolin::OpenGlRenderState* s_cam = new pangolin::OpenGlRenderState(
          //pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),
          pangolin::ProjectionMatrix(1024, 320, 420, 420, 512, 160, 0.1, 1000),
          pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 1, pangolin::AxisNegY));

      pangolin::View* v = new pangolin::View();

      v->SetBounds(std::get<0>(bound), std::get<1>(bound), std::get<2>(bound),
                   std::get<3>(bound), 1024.0/320.0)//640.0 / 480.0)
          .SetHandler(new pangolin::Handler3D(*s_cam));

      pangolin::Display("Map").AddDisplay(*v);
      for (auto context : rfs[i]->contexts()) {
        s_cams[context.second->id()] = s_cam;
        ctxIdToDisplay[context.second->id()] = v;
      }
    }
  }

  void preCall() {
    glClearColor(0.05 * !showcaseMode, 0.05 * !showcaseMode,
                 0.3 * !showcaseMode, 0.0f);
    if (showcaseMode) glClearColor(1.0, 1.0, 1.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    width = pangolin::DisplayBase().v.w;
    height = pangolin::DisplayBase().v.h;
  }

  void activateView(int id) { ctxIdToDisplay[id]->Activate(*(s_cams[id])); }

  inline void drawFrustum(const Eigen::Matrix4f& pose, const float scale = 0.1f) {
    // if(showcaseMode)
    //    return;

    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
    K(0, 0) = Intrinsics::getInstance().fx();
    K(1, 1) = Intrinsics::getInstance().fy();
    K(0, 2) = Intrinsics::getInstance().cx();
    K(1, 2) = Intrinsics::getInstance().cy();

    Eigen::Matrix3f Kinv = K.inverse();

    pangolin::glDrawFrustrum(Kinv, Resolution::getInstance().width(),
                             Resolution::getInstance().height(), pose, scale);
  }

  void displayImg(const std::string& id, GPUTexture* img) {
    // if(showcaseMode)
    //     return;

    glDisable(GL_DEPTH_TEST);

    pangolin::Display(id).Activate();
    img->texture->RenderToViewport(true);

    glEnable(GL_DEPTH_TEST);
  }

  void postCall() {
    GLint cur_avail_mem_kb = 0;
    glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX, &cur_avail_mem_kb);

    int memFree = cur_avail_mem_kb / 1024;

    gpuMem->operator=(memFree);

    pangolin::FinishFrame();

    glFinish();
  }

  void drawFXAA(pangolin::OpenGlMatrix mvp, pangolin::OpenGlMatrix mv,
                const std::pair<GLuint, GLuint>& model, const float threshold,
                const int time, const int timeIdx, const int timeDelta,
                const bool invertNormals) {
    // First pass computes positions, colors and normals per pixel
    colorFrameBuffer->Bind();

    glPushAttrib(GL_VIEWPORT_BIT);

    glViewport(0, 0, renderBuffer->width, renderBuffer->height);

    glClearColor(0.05 * !showcaseMode, 0.05 * !showcaseMode,
                 0.3 * !showcaseMode, 0);
    if (showcaseMode) glClearColor(1.0, 1.0, 1.0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    colorProgram->Bind();

    colorProgram->setUniform(Uniform("MVP", mvp));

    colorProgram->setUniform(Uniform("threshold", threshold));

    colorProgram->setUniform(Uniform("time", time));

    colorProgram->setUniform(Uniform("timeIdx", time));

    colorProgram->setUniform(Uniform("timeDelta", timeDelta));

    colorProgram->setUniform(Uniform("signMult", invertNormals ? 1.0f : -1.0f));

    colorProgram->setUniform(Uniform(
        "colorType", (drawNormals->Get()
                          ? 1
                          : drawColors->Get() ? 2 : drawTimes->Get() ? 3 : 0)));

    colorProgram->setUniform(Uniform("unstable", drawUnstable->Get()));

    colorProgram->setUniform(Uniform("drawWindow", drawWindow->Get()));

    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    // This is for the point shader
    colorProgram->setUniform(Uniform("pose", pose));

    Eigen::Matrix4f modelView = mv;

    Eigen::Vector3f lightpos = modelView.topRightCorner(3, 1);

    colorProgram->setUniform(Uniform("lightpos", lightpos));

    glBindBuffer(GL_ARRAY_BUFFER, model.first);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
        reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 1));

    int timeOffset = sizeof(Eigen::Vector4f) * 2;
    for (int i = 0; i < Vertex::MAX_SENSORS; i++) {
      glEnableVertexAttribArray(2 + i);
      glVertexAttribPointer(
          2 + i, 1, GL_FLOAT, GL_FALSE, Vertex::SIZE,
          reinterpret_cast<GLvoid*>(timeOffset + sizeof(float) * (i)));
    }

    glEnableVertexAttribArray(2 + Vertex::MAX_SENSORS);
    glVertexAttribPointer(
        2 + Vertex::MAX_SENSORS, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE,
        reinterpret_cast<GLvoid*>(timeOffset +
                                  sizeof(float) * Vertex::MAX_SENSORS));

    glDrawTransformFeedback(GL_POINTS, model.second);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    for (int i = 0; i < Vertex::MAX_SENSORS; i++) {
      glDisableVertexAttribArray(2 + i);
    }
    glDisableVertexAttribArray(2 + Vertex::MAX_SENSORS);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    colorFrameBuffer->Unbind();

    colorProgram->Unbind();

    glPopAttrib();

    fxaaProgram->Bind();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, colorTexture->texture->tid);

    Eigen::Vector2f resolution(renderBuffer->width, renderBuffer->height);

    fxaaProgram->setUniform(Uniform("tex", 0));
    fxaaProgram->setUniform(Uniform("resolution", resolution));

    glDrawArrays(GL_POINTS, 0, 1);

    fxaaProgram->Unbind();

    glBindFramebuffer(GL_READ_FRAMEBUFFER, colorFrameBuffer->fbid);

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    glBlitFramebuffer(0, 0, renderBuffer->width, renderBuffer->height, 0, 0,
                      width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, 0);

    glFinish();
  }

  void addCamera(std::string camName) {
    bool active = m_cameras.size() >= 1 ? false : true;
    m_activeCamera = active ? camName : m_activeCamera;
    pangolin::Var<bool>* label =
        new pangolin::Var<bool>("ui." + camName, active, true);
    m_cameras.push_back({camName, label});
    RegisterGuiVarChangedCallback(&GUI::activeLogReaderChangedCallback,
                                  (void*)this, "ui." + camName);
  }

  void clearLogReaders() { m_cameras.clear(); }

  static void activeLogReaderChangedCallback(void* data,
                                             const std::string& name,
                                             pangolin::VarValueGeneric& var) {
    GUI* thisptr = (GUI*)data;

    for (auto c : thisptr->m_cameras) {
      if (c.second->Meta().full_name.compare(name) != 0) {
        *(c.second) = false;
      } else {
        thisptr->m_activeCamera = c.first;
      }
    }
  }

  std::string& activeCamera() { return m_activeCamera; }

  bool showcaseMode;
  int width;
  int height;
  int panel;

  pangolin::Var<bool> *pause, *step, *save, *save_images, *reset, *flipColors, *rgbOnly,
      *pyramid, *so3, *frameToFrameRGB, *fastOdom, *followPose, *drawRawCloud,
      *drawFilteredCloud, *drawNormals, *autoSettings, *drawDefGraph,
      *drawColors, *drawFxaa, *drawGlobalModel, *drawUnstable, *drawPoints,
      *drawTimes, *drawFerns, *drawDeforms, *drawWindow, *drawContributions,
      *batchAlign;

  std::vector<std::pair<std::string, pangolin::Var<bool>*>> m_cameras;
  std::string m_activeCamera;

  pangolin::Var<int> *gpuMem, *numKfs, *numBinsImg, *numBinsDepth, * nidPyramidLevel;
  pangolin::Var<std::string> *totalPoints, *totalNodes, *totalFerns, *totalDefs,
      *totalFernDefs, *trackInliers, *trackRes, *logProgress;
  pangolin::Var<float> *confidenceThreshold, *depthCutoff, *icpWeight,
      *nidThreshold, *nidDepthWeight;

  pangolin::DataLog resLog, inLog, miLog;
  pangolin::Plotter *resPlot, *inPlot, *miPlot;

  pangolin::OpenGlRenderState s_cam;

  std::map<int, pangolin::OpenGlRenderState*> s_cams;
  std::map<int, pangolin::View*> ctxIdToDisplay;

  pangolin::GlRenderBuffer* renderBuffer;
  pangolin::GlFramebuffer* colorFrameBuffer;
  GPUTexture* colorTexture;
  std::shared_ptr<Shader> colorProgram;
  std::shared_ptr<Shader> fxaaProgram;
};

#endif /* GUI_H_ */
