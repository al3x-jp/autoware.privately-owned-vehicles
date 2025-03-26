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

#define SHOW_DEBUG 1

/*
**
FUNC:   read_pt_data_and_write_to_mem(..)
DESC:   Function to read in pretrained data and write to associated DNNL memory structure.
INPT:   File Name, DNNL Memory descriptor
**
*/
void read_pt_data_and_write_to_mem (    std::string const   &name, 
                                        dnnl::memory        &memory
                                    )
{
    // Construct filename
    std::ostringstream oss;
    oss << "../pretrained-data/" << name;
    std::string data_file = oss.str();

    std::cout << "INFO: Reading data file - " << data_file << std::endl;

    // Read data in
    std::ifstream input_file(data_file, std::ios_base::binary | std::ios_base::in);
    if (input_file.is_open()==false) throw std::runtime_error("ERROR - Failed to open file");

    std::vector<char> pretrained_data {std::istreambuf_iterator<char> {input_file}, {}};

    // Write data into memory
    write_to_dnnl_memory(pretrained_data.data(), memory);

    std::cout << "INFO: Data written to DNNL memory." << std::endl;
}

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
    std::cout << "INFO: Done." << std::endl;

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
        Operation Name:     "conv_0"
        Data Module Name:   "backbone.encoder.0.0"
    **
    */


    // Initialise Attributes
    dnnl::memory::dims conv_src_tz =     {batch, 3, 320, 640};
    dnnl::memory::dims conv_weights_tz = {32, 3, 3, 3};
    dnnl::memory::dims conv_dst_tz =     {batch, 32, 160, 320};
    dnnl::memory::dims conv_strides =    {2, 2};
    dnnl::memory::dims conv_padding =    {1, 1, 1, 1};

    // Allocate Buffers
    std::vector<float> conv_weights(product(conv_weights_tz));

    // Create Memory
    auto user_src_memory = dnnl::memory({{conv_src_tz}, dt::f32, tag::nchw}, dnn_engine);
    write_to_dnnl_memory(fPhysicallyPermutedInputArray, user_src_memory);

    auto user_weights_memory = dnnl::memory({{conv_weights_tz}, dt::f32, tag::oihw}, dnn_engine);
    write_to_dnnl_memory(conv_weights.data(), user_weights_memory);

    read_pt_data_and_write_to_mem   (   "backbone.encoder.0.0.weight.bin", 
                                        user_weights_memory
                                    );


    // Create Descriptors
    auto conv_src_md =     dnnl::memory::desc({conv_src_tz}, dt::f32, tag::any);
    auto conv_weights_md = dnnl::memory::desc({conv_weights_tz}, dt::f32, tag::any);
    auto conv_dst_md =     dnnl::memory::desc({conv_dst_tz}, dt::f32, tag::any);

    // Create Convolution Primitive
    auto conv_prim_desc = dnnl::convolution_forward::primitive_desc  (     dnn_engine,
                                                                            prop_kind::forward_inference, 
                                                                            algorithm::convolution_direct,
                                                                            conv_src_md, 
                                                                            conv_weights_md, 
                                                                            conv_dst_md,
                                                                            conv_strides, 
                                                                            conv_padding, 
                                                                            conv_padding
                                                                        );

    // Data Format Checks                                                                
    auto conv_src_memory = user_src_memory;

    if (conv_prim_desc.src_desc() != user_src_memory.get_desc()) 
    {
        conv_src_memory = dnnl::memory(conv_prim_desc.src_desc(), dnn_engine);
        dnn_scene_seg_net.push_back(dnnl::reorder(user_src_memory, conv_src_memory));
        dnn_scene_seg_net_args.push_back({{DNNL_ARG_FROM, user_src_memory},{DNNL_ARG_TO, conv_src_memory}});
    }

    auto conv_weights_memory = user_weights_memory;

    if (conv_prim_desc.weights_desc() != user_weights_memory.get_desc()) 
    {
        conv_weights_memory = dnnl::memory(conv_prim_desc.weights_desc(), dnn_engine);
        dnnl::reorder(user_weights_memory, conv_weights_memory).execute(dnn_stream, user_weights_memory, conv_weights_memory);
    }

    // Create output memory
    auto conv_dst_memory =dnnl::memory(conv_prim_desc.dst_desc(), dnn_engine);

    /// Create a convolution primitive and add it to the network container
    dnn_scene_seg_net.push_back(convolution_forward(conv_prim_desc));

    dnn_scene_seg_net_args.push_back(   {
                                            {DNNL_ARG_SRC, conv_src_memory},
                                            {DNNL_ARG_WEIGHTS, conv_weights_memory},
                                            {DNNL_ARG_DST, conv_dst_memory}
                                        }
                                    );


    /*
    **
        Operation Name:     "n_0"
        Data Module Name:   "backbone.encoder.0.1.bias.bin"
                            "backbone.encoder.0.1.weight.bin"
                            "backbone.encoder.0.1.running_mean.bin"
                            "backbone.encoder.0.1.running_var.bin"
    **
    */

    float bn_eps = 1.0e-5;

    // Configuring dimensions
    dnnl::memory::dims bn_src_dims = {batch, 32, 160, 320};
    dnnl::memory::dims bn_scaleshift_dims = {32};
    dnnl::memory::dims bn_mean_dims = {32};
    dnnl::memory::dims bn_var_dims = {32};

    dnnl::memory::format_tag bn_format = dnnl::memory::format_tag::nhwc;
    dnnl::memory::data_type bn_data_type = dnnl::memory::data_type::f32;

    // Create memory descriptors
    auto src_md = dnnl::memory::desc(bn_src_dims, bn_data_type, bn_format);
    auto dst_md = dnnl::memory::desc(bn_src_dims, bn_data_type, bn_format);

    auto bn_scaleshift_md = dnnl::memory::desc(bn_scaleshift_dims, bn_data_type, dnnl::memory::format_tag::a);
    auto bn_mean_md = dnnl::memory::desc(bn_mean_dims, bn_data_type, dnnl::memory::format_tag::x);
    auto bn_variance_md = dnnl::memory::desc(bn_var_dims, bn_data_type, dnnl::memory::format_tag::x);

    // Create memory
    //this->out_mem_ = dnnl::memory(dst_desc, dnn_engine);
    auto bn_scale_mem = dnnl::memory(bn_scaleshift_md, dnn_engine);
    auto bn_shift_mem = dnnl::memory(bn_scaleshift_md, dnn_engine);
    auto bn_mean_mem = dnnl::memory(bn_mean_md, dnn_engine);
    auto bn_variance_mem = dnnl::memory(bn_variance_md, dnn_engine);

    read_pt_data_and_write_to_mem   (   "backbone.encoder.0.1.bias.bin", 
        bn_shift_mem
    );
    read_pt_data_and_write_to_mem   (   "backbone.encoder.0.1.weight.bin", 
        bn_scale_mem
    );
    read_pt_data_and_write_to_mem   (   "backbone.encoder.0.1.running_mean.bin", 
        bn_mean_mem
    );
    read_pt_data_and_write_to_mem   (   "backbone.encoder.0.1.running_var.bin", 
        bn_variance_mem
    );

    // Set flags for bnorm
    dnnl::normalization_flags bn_flags = (dnnl::normalization_flags::use_scale
            | dnnl::normalization_flags::use_shift
            | dnnl::normalization_flags::use_global_stats);

    // No ReLU
    //if (_relu) flags |= dnnl::normalization_flags::fuse_norm_relu;

    auto batch_norm_pd = dnnl::batch_normalization_forward::primitive_desc  (   dnn_engine, 
                                                                                dnnl::prop_kind::forward_inference, 
                                                                                src_md,
                                                                                dst_md, 
                                                                                bn_eps, 
                                                                                bn_flags
                                                                            );

    // Create output memory
    auto bn_dst_memory =dnnl::memory(dst_md, dnn_engine);

    /// Create a convolution primitive and add it to the network container
    dnn_scene_seg_net.push_back(batch_normalization_forward(batch_norm_pd));

    dnn_scene_seg_net_args.push_back(   {
                                            {DNNL_ARG_SRC, conv_dst_memory},
                                            {DNNL_ARG_MEAN, bn_mean_mem},
                                            {DNNL_ARG_VARIANCE, bn_variance_mem},
                                            {DNNL_ARG_SCALE, bn_scale_mem},
                                            {DNNL_ARG_SHIFT, bn_shift_mem},
                                            {DNNL_ARG_DST, bn_dst_memory}
                                        }
                                    );            
                                    
    /*
    ** 
        Operation Name:     "SiLU"
                            "Sigmoid --> Multiply"

        SiLu often defined as x * sigmoid(x).
        OneDNN implements swish(alpha, x) = x * sigmoid(alpha * x). 
        In other words SiLu(x) = swish(1, x).
        
        A swish pattern is composed by a sigmoid op and a multiply op:

                any
               /   \
          sigmoid   |
               \   /
              multiply
                 |
                any
    **
    */

    // Tensor dimensions.
    const memory::dim   sw_N = 1,          // batch size
                        sw_IC = 32,        // channels
                        sw_IH = 160,       // tensor height
                        sw_IW = 320;       // tensor width

    // Source (src) and destination (dst) tensors dimensions.
    memory::dims sw_src_dims = {sw_N, sw_IC, sw_IH, sw_IW};
    memory::dims sw_dst_dims = {sw_N, sw_IC, sw_IH, sw_IW};

    // Allocate buffers. In this example, out-of-place primitive execution is
    // demonstrated since both src and dst are required for later backward
    // propagation.
    //std::vector<float> src_data(product(src_dims));
    std::vector<float> dst_data(product(sw_dst_dims));

    // Create src and dst memory descriptors and memory objects.
    auto sw_src_md = memory::desc(
        sw_src_dims, memory::data_type::f32, memory::format_tag::nchw);

    auto sw_dst_md = memory::desc(
        sw_dst_dims, memory::data_type::f32, memory::format_tag::nchw);

    //auto src_mem = memory(src_md, engine);
    auto sw_dst_mem = memory(sw_dst_md, dnn_engine);

    // Create primitive descriptor.
    auto sw_eltwise_pd = eltwise_forward::primitive_desc(  dnn_engine,   
                                                        prop_kind::forward, 
                                                        algorithm::eltwise_swish, 
                                                        sw_src_md,
                                                        sw_dst_md, 
                                                        1.0f,            // Alpha?
                                                        0.f
                                                    );

    /// Create a convolution primitive and add it to the network container
    dnn_scene_seg_net.push_back(eltwise_forward(sw_eltwise_pd));
    dnn_scene_seg_net_args.push_back(   {
                                            {DNNL_ARG_SRC, bn_dst_memory},
                                            {DNNL_ARG_DST, sw_dst_mem}
                                        }
                                    );   
    /*
    **
    EXECUTE NETWORK
    **
    */
    std::cout << "INFO: Running SceneSeg Network" << std::endl;

    assert(dnn_scene_seg_net.size() == dnn_scene_seg_net_args.size() && "ERROR - Something is missing!");

    for (size_t i = 0; i < dnn_scene_seg_net.size(); ++i)
    {
        std::cout << "INFO: Executing SceneSeg Layer <" << i << ">" << std::endl;
        dnn_scene_seg_net.at(i).execute(dnn_stream, dnn_scene_seg_net_args.at(i));
    }

    dnn_stream.wait();

#if SHOW_DEBUG

    // Temporary, read back output tensor data
    std::cout << "INFO: Reading back conv data" << std::endl;
    std::vector<float> output_conv(product(conv_dst_tz));
    read_from_dnnl_memory(output_conv.data(), conv_dst_memory);
    std::cout << std::endl << output_conv << std::endl << std::endl;

    // Temporary, read back output tensor data
    std::cout << "INFO: Reading back bn data" << std::endl;
    std::vector<float> output_bn(product(bn_src_dims));
    read_from_dnnl_memory(output_bn.data(), bn_dst_memory);
    std::cout << output_bn << std::endl << std::endl;

    // Temporary, read back output tensor data
    std::cout << "INFO: Reading back sw data" << std::endl;
    std::vector<float> output_sw(product(sw_dst_dims));
    read_from_dnnl_memory(output_sw.data(), sw_dst_mem);
    std::cout << output_sw << std::endl << std::endl;

#endif

    std::cout << "\033[1;33m" << "<== Program End" << "\033[0m\n" << std::endl << std::endl;
    
    // Done
    return 0;
}