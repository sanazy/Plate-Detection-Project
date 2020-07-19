#include <setjmp.h>
#include <stdio.h>
#include <string.h>

#include <cmath>
#include <fstream>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using tensorflow::uint8;


// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name, 
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '", graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}


// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(const string& file_name, 
                               const int input_height, const int input_width, 
                               std::vector<Tensor>* out_tensors,
							   const int pad_h, const int pad_w) {

  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string original_name = "identity";
  string output_name = "normalized";
  auto file_reader =
      tensorflow::ops::ReadFile(root.WithOpName(input_name), file_name);
  
  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  tensorflow::Output image_reader;
  image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader, 
                            DecodeJpeg::Channels(wanted_channels));

  // Also return identity so that we can know the original dimensions and
  // optionally save the image out with bounding boxes overlaid.
  auto original_image = Identity(root.WithOpName(original_name), image_reader);
  
  // Now cast the image data to float so we can do normal math on it.
  auto float_caster = Cast(root.WithOpName("float_caster"), original_image,
                           tensorflow::DT_FLOAT);
				
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root, float_caster, 0);
  
  // Bilinearly resize the image to fit the required dimensions.
  auto resized = ResizeBilinear(
      root, dims_expander,
      Const(root.WithOpName("size"), {input_height, input_width}));

  // Subtract the mean and divide by the scale.
  //Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
  //  {input_std});
  //Div(root.WithOpName(output_name), Sub(root, resized, {124.0f, 116.0f, 103.0f}),
   //   {58.0f, 57.0f, 57.0f});
	  
  auto normalized = Div(root.WithOpName("scale_normalization"), Sub(root, resized, {124.0f, 116.0f, 103.0f}),
     {58.0f, 57.0f, 57.0f});

  Pad(root.WithOpName(output_name), normalized, {{0, 0}, {0, pad_h},{0, pad_w},{0, 0}});
  
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(
      session->Run({}, {output_name, original_name}, {}, out_tensors));
  return Status::OK();
}


Status SaveImage(const Tensor& tensor, const string& file_path) {
  LOG(INFO) << "Saving image to " << file_path;
  CHECK(tensorflow::str_util::EndsWith(file_path, ".png"))
      << "Only saving of png files is supported.";

  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string encoder_name = "encode";
  string output_name = "file_writer";

  tensorflow::Output image_encoder =
      EncodePng(root.WithOpName(encoder_name), tensor);
  tensorflow::ops::WriteFile file_saver = tensorflow::ops::WriteFile(
      root.WithOpName(output_name), file_path, image_encoder);

  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  std::vector<Tensor> outputs;
  TF_RETURN_IF_ERROR(session->Run({}, {}, {output_name}, &outputs));

  return Status::OK();
}

// Analyzes the output of the MultiBox graph to retrieve the highest scores and
// their positions in the tensor, which correspond to individual box detections.
Status GetTopDetections(const std::vector<Tensor>& outputs, int how_many_labels,
                        Tensor* indices, Tensor* scores) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string output_name = "top_k";
  TopK(root.WithOpName(output_name), outputs[0], how_many_labels);
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensors.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  // The TopK node returns two outputs, the scores and their original indices,
  // so we have to append :0 and :1 to specify them both.
  std::vector<Tensor> out_tensors;
  TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"},
                                  {}, &out_tensors));
  *scores = out_tensors[0];
  *indices = out_tensors[1];
  return Status::OK();
}

// Converts an encoded location to an actual box placement
void DecodeLocation(const float* encoded_location,
                    float* decoded_location) {
  bool non_zero = false;
  for (int i = 0; i < 4; ++i) {
    const float curr_encoding = encoded_location[i];
    non_zero = non_zero || curr_encoding != 0.0f;
    decoded_location[i] = curr_encoding;
  }

  if (!non_zero) {
    LOG(WARNING) << "No non-zero encodings; check log for inference errors.";
  }
}

float DecodeScore(float encoded_score) {
  return 1 / (1 + std::exp(-encoded_score));
}


void DrawBox(const int image_width, const int image_height, int left, int top,
             int right, int bottom, tensorflow::TTypes<uint8>::Flat* image) {
  tensorflow::TTypes<uint8>::Flat image_ref = *image;

  top = std::max(0, std::min(image_height - 1, top));
  bottom = std::max(0, std::min(image_height - 1, bottom));

  left = std::max(0, std::min(image_width - 1, left));
  right = std::max(0, std::min(image_width - 1, right));
  
  for (int i = 0; i < 3; ++i) {
    uint8 val = i == 2 ? 255 : 0;
    for (int x = left; x <= right; ++x) {
      image_ref((top * image_width + x) * 3 + i) = val;
      image_ref((bottom * image_width + x) * 3 + i) = val;
    }
    for (int y = top; y <= bottom; ++y) {
      image_ref((y * image_width + left) * 3 + i) = val;
      image_ref((y * image_width + right) * 3 + i) = val;
    }
  }
}


// Given the output of a model run, and the name of a file containing the labels
// this prints out the top five highest-scoring values.
Status PrintTopDetections(const std::vector<Tensor>& outputs,
                          const int num_boxes,
                          const int num_detections,
                          const string& image_file_name,
                          Tensor* original_tensor, const int input_height, const int input_width) {
  std::vector<float> locations;

  const int how_many_labels = num_detections;
  Tensor indices;
  Tensor scores;
  TF_RETURN_IF_ERROR(
      GetTopDetections(outputs, how_many_labels, &indices, &scores));

  tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();

  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();

  const Tensor& encoded_locations = outputs[1];
  auto locations_encoded = encoded_locations.flat<float>();

  LOG(INFO) << original_tensor->DebugString();
  const int image_width = original_tensor->shape().dim_size(1);
  const int image_height = original_tensor->shape().dim_size(0);
  
  tensorflow::TTypes<uint8>::Flat image_flat = original_tensor->flat<uint8>();
 
  LOG(INFO) << "===== Top " << how_many_labels << " Detections ======";
  for (int pos = 0; pos < how_many_labels; ++pos) {
    const int label_index = indices_flat(pos);
    const float score = scores_flat(pos);

    float decoded_location[4];
    DecodeLocation(&locations_encoded(label_index * 4), decoded_location);

    
	//float left = decoded_location[0] * image_width;
    //float top = decoded_location[1] * image_height;
    //float right = decoded_location[2] * image_width;
    //float bottom = decoded_location[3] * image_height;
	
	float scale_w = image_width / 256;
	float scale_h = image_height / 256;
	
	float left = decoded_location[0] * scale_w;
    float top = decoded_location[1] * scale_h;
    float right = decoded_location[2] * scale_w;
    float bottom = decoded_location[3] * scale_h;

    LOG(INFO) << "Detection " << pos << ": "
              << "L:" << left << " "
              << "T:" << top << " "
              << "R:" << right << " "
              << "B:" << bottom << " "
              << "(" << label_index << ") score: " << DecodeScore(score);

    DrawBox(image_width, image_height, left, top, right, bottom, &image_flat);
  }

  if (!image_file_name.empty()) {
    return SaveImage(*original_tensor, image_file_name);
  }
  return Status::OK();
}


int main(int argc, char* argv[]) {
  // These are the command-line flags the program can understand.
  // They define where the graph and input data is located, and what kind of
  // input the model expects. If you train your own model, or use something
  // other than multibox_model you'll need to update these.
  string image =
      "tensorflow/examples/multibox_detector/data/car.jpg";
  string image2 =
      "tensorflow/examples/multibox_detector/data/car2.jpg";
  string graph =
      "tensorflow/examples/multibox_detector/data/model.pb";

  int32 input_width = 256;
  int32 input_height = 256;
  int32 input_size = 256;
  int32 num_detections = 5;
  int32 num_boxes = 784;
  
  string input_layer = "input_1:0";
  string output_boxes_layer = "filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0";
  string output_scores_layer = "filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0";
  string output_labels_layer = "filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0";

  string root_dir = "";
  string image_out = "";
  
  std::vector<Flag> flag_list = {
      Flag("image", &image, "image to be processed"),
	  Flag("image2", &image2, "image2 to be processed"),
	  Flag("graph", &graph, "graph to be executed"),
      Flag("image_out", &image_out, "location to save output image, if desired"),
	  Flag("root_dir", &root_dir, "interpret image and graph file names relative to this directory"),
	  
      Flag("input_width", &input_width, "resize image to this width in pixels"),
      Flag("input_height", &input_height, "resize image to this height in pixels"),
      Flag("num_detections", &num_detections, "number of top detections to return"),
      Flag("num_boxes", &num_boxes, "number of boxes defined by the location file"),
		   
      Flag("input_layer", &input_layer, "name of input layer"),
      Flag("output_boxes_layer", &output_boxes_layer, "name of location output layer"),
      Flag("output_scores_layer", &output_scores_layer, "name of score output layer"),
	  Flag("output_labels_layer", &output_labels_layer, "name of labels output layer"),
		   
  };
  
  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }
  
  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  } 
  
  // First we load and initialize the model.
  std::unique_ptr<tensorflow::Session> session;
  string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }
  
  // Get the image from disk as a float array of numbers, resized and normalized
  // to the specifications the main graph expects.
  std::vector<Tensor> image_tensors;
  string image_path = tensorflow::io::JoinPath(root_dir, image);

  Status read_tensor_status =
      ReadTensorFromImageFile(image_path, input_height, input_width, &image_tensors, 0, 0);
  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
    return -1;
  }
  const Tensor& resized_tensor = image_tensors[0];
   	
  LOG(INFO) << "Starting ...";
  
  // Actually run the image through the model.
  std::vector<Tensor> outputs;
  Status run_status = session->Run({{input_layer, resized_tensor}}, 
                                   {output_scores_layer, output_boxes_layer}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }
  
  LOG(INFO) << "Finished.";
  
  Status print_status = PrintTopDetections(outputs, num_boxes, num_detections, image_out, 
                                           &image_tensors[1], input_height, input_width);

  if (!print_status.ok()) {
    LOG(ERROR) << "Running print failed: " << print_status;
    return -1;
  }
  
  
  ////////////////////////////////
  
  // Get the image from disk as a float array of numbers, resized and normalized
  // to the specifications the main graph expects.
  std::vector<Tensor> image_tensors2;
  string image_path2 = tensorflow::io::JoinPath(root_dir, image2);

  Status read_tensor_status2 =
      ReadTensorFromImageFile(image_path2, input_height, input_width, &image_tensors2, 0, 0);
  if (!read_tensor_status2.ok()) {
    LOG(ERROR) << read_tensor_status2;
    return -1;
  }
  const Tensor& resized_tensor2 = image_tensors2[0];
   	
  LOG(INFO) << "Starting ...";
  
  // Actually run the image through the model.
  std::vector<Tensor> outputs2;
  Status run_status2 = session->Run({{input_layer, resized_tensor2}}, 
                                   {output_scores_layer, output_boxes_layer}, {}, &outputs2);
  if (!run_status2.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }
  
  LOG(INFO) << "Finished.";
  
  Status print_status2 = PrintTopDetections(outputs2, num_boxes, num_detections, image_out, 
                                           &image_tensors[1], input_height, input_width);

  if (!print_status2.ok()) {
    LOG(ERROR) << "Running print failed: " << print_status;
    return -1;
  }
  
  
  
  return 0;
}
