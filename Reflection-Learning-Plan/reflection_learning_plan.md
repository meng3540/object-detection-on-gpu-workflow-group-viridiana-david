# Reflection and Learning Plan

## Reflection

During this project, we learned how object detection can be used on a GPU edge device. This project helped us understand more how AI inference is related to parallel programming, because the image has many pixels and many operations that need to be processed.

We used tools such as Python, OpenCV, PyTorch, CUDA, Jetson containers, and YOLOv8. We also learned that the setup of the environment is very important. If one tool is not installed correctly, the program may not run as expected or the GPU may not be detected.

One thing that went well was the live object detection test. The camera was able to open, and the program detected objects with their corresponding boxes, so that means that the model was able to identify the objects, funny enough, to know that Yolov8n has a label for objects like ties. When the program was running on the GPU, the video was smoother. When it was running on the CPU, the video was more laggy, so this helped us see the difference between CPU and GPU inference.

One challenge was the first approach with TensorFlow. At the beginning, TensorFlow was considered for the workflow, but it became harder because of compatibility issues and setup problems. Because of this, we decided to use YOLOv8 with PyTorch, which was easier to implement for this project.

Another challenge was making the GPU run correctly. At first, CUDA was not working properly, so some tools had to be uninstalled and installed again. This part was important because the project needed to show that GPU acceleration was being used.

## Learning Plan

This project showed us that we still need more practice with AI models running on embedded devices. We were able to make the workflow work, but there are still more topics that we need to understand better, such as AI neural networks and their implementation, including training models.

In the future, we would also like to learn more about TensorFlow because it was our first option for this project, but we were not able to use it successfully. Learning TensorFlow better would help us understand another important AI framework and give us more options for future projects.

If this project were continued in the future, we would need to learn more about how to make the detection more accurate and stable. This could include training the model with our own images, testing it in different environments, and improving how the camera and lighting affect the results.

For future learning, useful resources would be NVIDIA Jetson documentation, Ultralytics YOLO documentation, PyTorch tutorials, OpenCV documentation, and TensorFlow tutorials.

Overall, this project gave us more confidence working with AI tools on Jetson devices, and it showed us that setting up the environment correctly is basically the hardest part of the project and it is as important as writing the code itself.
