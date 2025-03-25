/*
**
FILE:   sycl_main.cpp
DESC:
DESC:   oneDNN implementation of exported "SceneSeg" Network 
DESC:   - https://github.com/autowarefoundation/autoware.privately-owned-vehicles
DESC:
INFO:   Based on/modified versions of:      "pointnet" from here: "oneDNN/examples/network/pointnet.cpp", and
INFO:                                       "cnn_inference_f32.cpp" from here: "oneDNN/examples/cnn_inference_f32.cpp"           
INFO:
INFO:   This file is dependant on the following file: "oneDNN/examples/example_utils.hpp" from the oneDNN repository.
INFO:                          
INFO:   Other Dependancies: libTORCH C++ and OpenCV        
**
*/

// Standard Headers
#include <array>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <unordered_map>

// oneDNN
#include <oneapi/dnnl/dnnl.hpp>

// libTorch
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/cuda.h>

//OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// oneDNN Helper Functions (included from: oneDNN/examples/example_utils.hpp )
#include "example_utils.hpp"

using namespace dnnl;
using namespace cv; 
using namespace std;

/*
**
FUNC:   main()
DESC:   example program entry point
**
*/
int main(int argc, char *argv[]) 
{
    using tag = memory::format_tag;
    using dt = memory::data_type;

    std::cout << std::endl << "\033[1;33m" << "==> Program Start:" << "\033[0m\n"  << std::endl;

    if (argc != 3) {
        std::cerr << "Usage: ./onednn_scene_seg <path-to-pretrained-data-dir> <path-to-input-image-file>" << std::endl;
        return -1;
    }

    /*
    **
    Use libTORCH and OpenCV to load and prepare the input image
    **
    */

    // Load an input image
    cv::Mat frame = cv::imread(argv[2], cv::IMREAD_COLOR); 
    cv::Mat preImg;
    cvtColor(frame, preImg, cv::COLOR_BGR2RGB);

    // Resize
    cv::Mat img;
    resize(preImg, img, Size(640, 320));    // Fixed ratio from original network implementation

    // Convert input cv image to Tensor
    at::Tensor tensor_image = torch::from_blob(img.data, { img.rows, img.cols, 3 }, at::kByte);

    // Convert to float and scale it 
    tensor_image = tensor_image.toType(c10::kFloat).div(255);

    // Transpose the image
    tensor_image = tensor_image.permute({ (2),(0),(1) });

    // Create a batch dimension for input
    tensor_image.unsqueeze_(0);

    // Normalise the input values
    std::vector<double> norm_mean = {0.485, 0.456, 0.406};
    std::vector<double> norm_std = {0.229, 0.224, 0.225};
    tensor_image = torch::data::transforms::Normalize<>(norm_mean, norm_std)(tensor_image);

    // Need to physically permute the input data from RGBRGBRGB to RRRGGGBBB... etc
    float fPhysicallyPermutedInputArray[1][3][320][640];

    std::cout << "INFO: Permuting Input Data:" << std::endl;
    for (int chan=0;chan<3;chan++)
    {
        for (int cols=0;cols<640;cols++)
        {
            for (int rows=0;rows<320;rows++)
            {
                fPhysicallyPermutedInputArray[0][chan][rows][cols] = tensor_image[0][chan][rows][cols].item<float>();
            }
        }
    }
    std::cout << "INFO: Done." << std::endl << std::endl;

    /*
    **
    Initialise oneDNN Engine
    **
    */
    dnnl::engine dnn_engine(engine::kind::gpu, 0);
    dnnl::stream dnn_stream(dnn_engine);

    // Create network container
    std::vector<primitive>                          dnn_scene_seg_net;
    std::vector<std::unordered_map<int, memory>>    dnn_scene_seg_net_args;

    // Always single batch at the moment
    const memory::dim batch = 1;

    /*
    **
    Start Creating Network Layers
    **
    */

    /*
    **
    SceneSeg: conv_0

    {batch, 3, 320, 640} (x) {32, 3, 3, 3} 
        -> {batch, 32, 150, 320}
    strides: {2, 2}
    **
    */

    // Initialise Attributes
    dnnl::memory::dims conv0_src_tz =     {batch, 3, 320, 640};
    dnnl::memory::dims conv0_weights_tz = {32, 3, 3, 3};
    dnnl::memory::dims conv0_dst_tz =     {batch, 32, 160, 320};
    dnnl::memory::dims conv0_strides =    {2, 2};
    dnnl::memory::dims conv0_padding =    {1, 1, 1, 1};

    // Allocate Buffers
    //std::vector<float> user_src(batch * 3 * 320 * 640);
    std::vector<float> user_dst(batch * 1000);
    std::vector<float> conv0_weights(product(conv0_weights_tz));

    // Create Memory
    auto user_src_memory = dnnl::memory({{conv0_src_tz}, dt::f32, tag::nchw}, dnn_engine);
    write_to_dnnl_memory(fPhysicallyPermutedInputArray, user_src_memory);

    auto user_weights_memory = dnnl::memory({{conv0_weights_tz}, dt::f32, tag::oihw}, dnn_engine);
    write_to_dnnl_memory(conv0_weights.data(), user_weights_memory);

    // Read Weights and Bias Data
    std::ostringstream oss;
    oss << argv[1] << "/backbone.encoder.0.0.weight.bin";
    std::string data_file = oss.str();
    std::cout << std::endl << "INFO: Reading - " << data_file << std::endl << std::endl;

    std::ifstream input_file(data_file, std::ios_base::binary | std::ios_base::in);
    if (input_file.is_open()==false) throw std::runtime_error("ERROR - Failed to open file");
    std::vector<char> weights {std::istreambuf_iterator<char> {input_file}, {}};

    // Write data into memory
    write_to_dnnl_memory(weights.data(), user_weights_memory);

    // Create Descriptors
    auto conv0_src_md =     dnnl::memory::desc({conv0_src_tz}, dt::f32, tag::any);
    auto conv0_weights_md = dnnl::memory::desc({conv0_weights_tz}, dt::f32, tag::any);
    auto conv0_dst_md =     dnnl::memory::desc({conv0_dst_tz}, dt::f32, tag::any);

    // Create Convolution Primitive
    auto conv0_prim_desc = dnnl::convolution_forward::primitive_desc  (     dnn_engine,
                                                                            prop_kind::forward_inference, 
                                                                            algorithm::convolution_direct,
                                                                            conv0_src_md, 
                                                                            conv0_weights_md, 
                                                                            conv0_dst_md,
                                                                            conv0_strides, 
                                                                            conv0_padding, 
                                                                            conv0_padding
                                                                        );

    // Data Format Checks                                                                
    auto conv0_src_memory = user_src_memory;

    if (conv0_prim_desc.src_desc() != user_src_memory.get_desc()) 
    {
        conv0_src_memory = dnnl::memory(conv0_prim_desc.src_desc(), dnn_engine);
        dnn_scene_seg_net.push_back(dnnl::reorder(user_src_memory, conv0_src_memory));
        dnn_scene_seg_net_args.push_back({{DNNL_ARG_FROM, user_src_memory},{DNNL_ARG_TO, conv0_src_memory}});
    }

    auto conv0_weights_memory = user_weights_memory;

    if (conv0_prim_desc.weights_desc() != user_weights_memory.get_desc()) 
    {
        conv0_weights_memory = dnnl::memory(conv0_prim_desc.weights_desc(), dnn_engine);
        dnnl::reorder(user_weights_memory, conv0_weights_memory).execute(dnn_stream, user_weights_memory, conv0_weights_memory);
    }

    // Create output memory
    auto conv0_dst_memory =dnnl::memory(conv0_prim_desc.dst_desc(), dnn_engine);

    /// Create a convolution primitive and add it to the network container
    dnn_scene_seg_net.push_back(convolution_forward(conv0_prim_desc));

    dnn_scene_seg_net_args.push_back(   {
                                            {DNNL_ARG_SRC, conv0_src_memory},
                                            {DNNL_ARG_WEIGHTS, conv0_weights_memory},
                                            {DNNL_ARG_DST, conv0_dst_memory}
                                        }
                                    );

    /*
    **
    END OF SceneSeg: conv_0
    **
    */

    /*
    **
    SceneSeg: n_0

    {batch, 3, 320, 640} (x) {32, 3, 3, 3} 
        -> {batch, 32, 150, 320}
    strides: {2, 2}
    **
    */

    /*
    **
    EXECUTE NETWORK
    **
    */
    std::cout << std::endl << "INFO: Running SceneSeg Network" << std::endl;

    assert(dnn_scene_seg_net.size() == dnn_scene_seg_net_args.size() && "ERROR - Something is missing!");

    for (size_t i = 0; i < dnn_scene_seg_net.size(); ++i)
    {
        std::cout << "INFO: Executing SceneSeg Layer <" << i << ">" << std::endl << std::endl;
        dnn_scene_seg_net.at(i).execute(dnn_stream, dnn_scene_seg_net_args.at(i));
    }

    dnn_stream.wait();

    // Temporary, read back output tensor data
    std::cout << std::endl << "INFO: Reading back output data" << std::endl;
    std::vector<float> output(product(conv0_dst_tz));
    read_from_dnnl_memory(output.data(), conv0_dst_memory);

    std::cout << std::endl << output << std::endl;

    std::cout << std::endl << "\033[1;33m" << "<== Program End" << "\033[0m\n" << std::endl << std::endl;
    
    // Done
    return 0;
}