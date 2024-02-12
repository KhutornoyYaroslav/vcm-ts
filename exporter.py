import os
import onnx
import torch
import logging
import argparse
import numpy as np
import onnxruntime as ort
from core.config import cfg
from core.utils import dist_util
from core.utils.logger import setup_logger
from core.modelling.model import build_model
from core.utils.checkpoint import CheckPointer


def load_model(cfg, args):
    logger = logging.getLogger('EXPORT')

    # Create device
    device = torch.device(cfg.MODEL.DEVICE)

    # Create model
    model = build_model(cfg)
    model.to(device)

    # Create checkpointer
    arguments = {"epoch": 0}
    save_to_disk = dist_util.is_main_process()
    checkpointer = CheckPointer(model, None, None, None, cfg.OUTPUT_DIR, save_to_disk, logger)
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    return model


def main() -> int:
    # Create argument parser
    parser = argparse.ArgumentParser(description='PyTorch Export To ONNX')
    parser.add_argument('--config-file', dest='config_file', type=str, default="outputs/ftvsr/cfg.yaml",
                        help="Path to config file")
    parser.add_argument('--chunk-size', dest='chunk_size', type=int, default=10,
                        help="Time dimension size for onnx model input")
    parser.add_argument('--tile-size', dest='tile_size', type=int, default=160,
                        help="Height and width dimensions size for onnx model input")
    parser.add_argument('--onnx-opset', dest="onnx_opset", type=int, default=18, # For BasicVSR = 16, For FTVSR = 18
                        help='Target onnx opset')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()

    NUM_GPUS = 1
    args.distributed = False
    args.num_gpus = NUM_GPUS

    # Enable cudnn auto-tuner to find the best algorithm to use for your hardware
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Create config
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Create logger
    logger = setup_logger("EXPORT", dist_util.get_rank())
    logger.info("Using {} GPUs".format(NUM_GPUS))
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))

    # Create output export dir
    folder_path = os.path.join(cfg.OUTPUT_DIR, "export/onnx/")
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    # Do export
    with torch.no_grad():
        # Prepare model input shape
        input_shape = (1, args.chunk_size, 3, args.tile_size, args.tile_size)

        # Prepare result filenames
        input_shape_str = "_n{0:d}_t{1:d}_c{2:d}_h{3:d}_w{4:d}".format(*input_shape)
        model_filename = os.path.join(folder_path, str(cfg.MODEL.ARCHITECTURE) + input_shape_str + ".onnx")
        model_imp_filename = os.path.join(folder_path, str(cfg.MODEL.ARCHITECTURE) + "_imp" + input_shape_str + ".onnx")

        # Load model
        torch.cuda.empty_cache()
        device = torch.device(cfg.MODEL.DEVICE)
        model = load_model(cfg, args)
        model.eval()

        # Export model to onnx
        #
        # NOTE: Only for 'FTVSR' model architectures
        #
        # NOTE: Before exporting you need to make a modification to the Torch sources.
        # Open the file torch/onnx/utils.py and locate the '_export' function.
        # Comment out the line that checks the ONNX proto using the following code:
        #
        #
        #       if (operator_export_type is _C_onnx.OperatorExportTypes.ONNX) and (
        #           not val_use_external_data_format
        #       ):
        #           try:
        #               pass #_C._check_onnx_proto(proto)
        #           except RuntimeError as e:
        #               raise errors.CheckerError(e) from e
        #
        #
        # NOTE: Do not forget to undo this change!

        inputs = torch.randn(size=input_shape, dtype=torch.float32).to(device)
        torch.onnx.export(model,
                          inputs,
                          model_filename,
                          verbose=False, 
                          opset_version=args.onnx_opset,
                          keep_initializers_as_inputs = False,
                          input_names=["lrs"],
                          output_names=["out"]
        )

        # Fix exported model
        if cfg.MODEL.ARCHITECTURE in ['FTVSR']:
            onnx_model = onnx.load(model_filename)
            graph = onnx_model.graph
            for node in graph.node:
                if "ReduceMax" in node.op_type:
                    for index in range(len(node.attribute)):
                        if node.attribute[index].name == "axes":
                            del node.attribute[index]
                            axes_input = onnx.helper.make_tensor_value_info("axes", onnx.TensorProto.INT64, [1])
                            axes_value = onnx.numpy_helper.from_array(np.array([2]), "axes")
                            onnx_model.graph.input.extend([axes_input])
                            onnx_model.graph.initializer.extend([axes_value])
                            node.input.append("axes")
                            break
            onnx.save(onnx_model, model_filename)
            return 0

        # Validate model
        onnx_model = onnx.load(model_filename)
        onnx.checker.check_model(onnx_model)

        # Export imp model
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.optimized_model_filepath = model_imp_filename
        session = ort.InferenceSession(model_filename, so,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    return 0


if __name__ == '__main__':
    exit(main())
