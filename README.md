# adsmn-Face-cut-app
Face cut APP to remove face from images

#### Link to run app : http://ec2-34-232-53-179.compute-1.amazonaws.com:8501/

### Python Script

The main component of this repository is the Python script `predict.py`, which performs semantic segmentation and face processing on images.

#### Semantic Segmentation

The script utilizes the BiSeNet model for semantic segmentation. Here's how it works:

- **Model Initialization**: The `SemanticSegmentationModel` class initializes the BiSeNet model and loads pre-trained weights from the specified model file.
- **Image Preprocessing**: Before inference, the input image is preprocessed using transformations such as resizing and normalization.
- **Inference**: The `infer` method applies the model to the input image and produces a segmentation map.
- **Postprocessing**: The segmentation map is processed to extract facial regions and enhance their quality.

#### Face Processing

The script also includes functionality for processing detected faces within the segmented image. Here's what it does:

- **Face Detection**: The `FaceDetection` class detects faces within the input image using a separate face detection model.
- **Face Isolation**: After semantic segmentation, the script isolates facial regions from the segmented image based on the segmentation results.
- **Face Enhancement**: Further processing techniques are applied to improve the quality of the extracted faces.

### Installation

To use the script, follow these steps:

1. **Clone the Repository**: Clone this repository to your local machine.
2. **Install Dependencies**: Install the required dependencies listed in `requirements.txt`.

### Usage

Once installed, you can use the script as follows:

```bash
python app.py 
```
### Example

Here's an example of how to use the script:

```bash
python predict.py 
```

This command will process the `example.jpg` image and save the segmented and processed result as `output.jpg`.

### Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

### License

This project is licensed under the [MIT License](LICENSE).

---

This explanation provides users with a clear understanding of the script's functionality and how they can use it in their projects. Feel free to customize it further based on your specific code and requirements!
