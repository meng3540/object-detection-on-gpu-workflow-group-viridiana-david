# Performance Notes

## Testing Cases

The object detection program was tested in two cases: CPU-based inference and GPU-based inference.

## CPU-Based Inference

When the model was running on the CPU, the live detection was laggy. The video response was slower, and the detection window did not update as smoothly.

## GPU-Based Inference

When the model was running on the GPU, the detection was smoother. The camera feed responded better in real time, and the bounding boxes updated more consistently.

## Observed Output

The system detected objects from the live camera feed and displayed bounding boxes, class labels, and confidence values. Some detected objects included people, chairs, a mouse, and screens.

## Conclusion

The comparison showed that GPU acceleration improved the real-time performance of the object detection workflow, supporting the purpose of using a GPU edge device for AI inference.
