# inference
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--metrics ${METRICS}] [--out ${RESULT_FILE}]

example:
    python tools/test.py --config configs/mobilenet_v2/mobilenet-v2_8xb32_LaserLabel.py --checkpoint tools/work_dirs/mobilenet-v2_8xb32_LaserLabel/epoch_100.pth --metrics "accuracy" "precision" "recall" "f1_score" --out results/metrics.json
    python tools/test.py --config configs/mobilenet_v2/mobilenet-v2_8xb32_LaserLabel.py --checkpoint tools/work_dirs/mobilenet-v2_8xb32_LaserLabel/epoch_100.pth --out results/result.json

    python tools/test.py --config configs/resnet/resnet18_8xb16_LaserLabel.py --checkpoint tools/work_dirs/resnet18_8xb16_LaserLabel/epoch_100.pth --metrics "accuracy" "precision" "recall" "f1_score" --out experiment/metrics.json
    python tools/test.py --config configs/resnet/resnet18_8xb16_LaserLabel.py --checkpoint tools/work_dirs/resnet18_8xb16_LaserLabel/epoch_100.pth --out experiment/result.json

    python tools/test.py --config configs/mobilenet_v2/mobilenet-v2_8xb32_frameH500.py --checkpoint tools/work_dirs/mobilenet-v2_8xb32_frameH500/epoch_300.pth --out results/result.json

# transfer model
python tools/deployment/pytorch2torchscript.py \
    ${CONFIG_FILE} \
    --checkpoint ${CHECKPOINT_FILE} \
    --output-file ${OUTPUT_FILE} \
    --shape ${IMAGE_SHAPE} \
    --verify \

example:
  python tools/deployment/pytorch2torchscript.py configs/mobilenet_v2/mobilenet-v2_8xb32_LaserLabel.py --checkpoint tools/work_dirs/mobilenet-v2_8xb32_LaserLabel/epoch_100.pth --output-file experiment/LaserLabel_epoch100.pt --verify
  python tools/deployment/pytorch2onnx.py configs/mobilenet_v2/mobilenet-v2_8xb32_LaserLabel.py --checkpoint tools/work_dirs/mobilenet-v2_8xb32_LaserLabel/epoch_100.pth --output-file experiment/LaserLabel_epoch100.onnx --shape 180 240 --verify

  python tools/deployment/pytorch2torchscript.py configs/resnet/resnet18_8xb16_LaserLabel.py --checkpoint tools/work_dirs/resnet18_8xb16_LaserLabel/epoch_100.pth --output-file experiment/LaserLabel_epoch100.pt --verify
  python tools/deployment/pytorch2onnx.py configs/resnet/resnet18_8xb16_LaserLabel.py --checkpoint tools/work_dirs/resnet18_8xb16_LaserLabel/epoch_100.pth --output-file experiment/LaserLabel_epoch100.onnx --shape 180 240 --verify

  python tools/deployment/pytorch2torchscript.py configs/mobilenet_v2/mobilenet-v2_8xb32_frameH500.py --checkpoint tools/work_dirs/mobilenet-v2_8xb32_frameH500/epoch_500.pth --output-file experiment/frameH500_epoch500.pt --verify
  python tools/deployment/pytorch2onnx.py configs/mobilenet_v2/mobilenet-v2_8xb32_frameH500.py --checkpoint tools/work_dirs/mobilenet-v2_8xb32_frameH500/epoch_300.pth --output-file experiment/ColorLine300_epoch300.onnx --shape 192 256 --verify

# remove onnxruntime warning
  python remove_initializer_from_input.py --input experiment/LaserLabel_epoch100.onnx --output experiment/LaserLabel_epoch100.onnx
  python remove_initializer_from_input.py --input experiment/frameH500_epoch500.onnx --output experiment/frameH500_epoch500.onnx
  python remove_initializer_from_input.py --input experiment/ColorLine300_epoch300.onnx --output experiment/ColorLine300_epoch300.onnx

# visualization
  python tools/visualizations/vis_pipeline.py \
      ${CONFIG_FILE} \
      [--output-dir ${OUTPUT_DIR}] \
      [--phase ${DATASET_PHASE}] \
      [--number ${BUNBER_IMAGES_DISPLAY}] \
      [--skip-type ${SKIP_TRANSFORM_TYPE}] \
      [--mode ${DISPLAY_MODE}] \
      [--show] \
      [--adaptive] \
      [--min-edge-length ${MIN_EDGE_LENGTH}] \
      [--max-edge-length ${MAX_EDGE_LENGTH}] \
      [--bgr2rgb] \
      [--window-size ${WINDOW_SIZE}] \
      [--cfg-options ${CFG_OPTIONS}]
  python tools/visualizations/vis_pipeline.py configs/mobilenet_v2/mobilenet-v2_8xb32_frameH500.py --output-dir tmp --mode pipeline --number 1 --show --adaptive
