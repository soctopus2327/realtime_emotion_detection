import Webcam from "react-webcam";
import { useFaceDetection } from "react-use-face-detection";
import FaceDetection from "@mediapipe/face_detection";
import { Camera } from "@mediapipe/camera_utils";
import { useEffect, useRef, useState } from "react";
import * as R from "ramda";
import * as tf from "@tensorflow/tfjs";
import "../stylesheet/WebcamModified.css";

const cameraDisplayStyle = {
  width: "60%",
  position: "absolute",
};

const emotions = [
  "😡 angry : ",
  "🤮 disgust : ",
  "😨 fear : ",
  "😄 happy : ",
  "😐 neutral : ",
  "😭 sad : ",
  "😯 surprise : ",
];

const link =
  "https://raw.githubusercontent.com/Im-Rises/emotion-recognition-website/main/resnet50js_ferplus/model.json";

const drawOnCanvas = (context, video, boundingBox, emotionRecognizer) => {
  context.clearRect(0, 0, context.canvas.width, context.canvas.height);

  context.drawImage(video, 0, 0, context.canvas.width, context.canvas.height);

  // recuperation of all values into boudingBox (coordinate of face)
  const xCenterBoundingBox = R.pluck("xCenter", boundingBox);
  const yCenterBoundingBox = R.pluck("yCenter", boundingBox);
  const widthBoundingBox = R.pluck("width", boundingBox);
  const heightBoundingBox = R.pluck("height", boundingBox);

  // rectangle draw all around the face
  context.beginPath();
  context.lineWidth = "0.8";
  context.strokeStyle = "red";
  context.rect(
    xCenterBoundingBox * context.canvas.width,
    yCenterBoundingBox * context.canvas.height,
    widthBoundingBox * context.canvas.width,
    heightBoundingBox * context.canvas.height
  );
  context.stroke();

  // recuperation of face only if boundingBox has valuable coordinates
  if (
    xCenterBoundingBox > 0 &&
    yCenterBoundingBox > 0 &&
    widthBoundingBox > 0 &&
    heightBoundingBox > 0
  ) {
    let face = context.getImageData(
      xCenterBoundingBox * context.canvas.width,
      yCenterBoundingBox * context.canvas.height,
      widthBoundingBox * context.canvas.width,
      heightBoundingBox * context.canvas.height
    );
    if (typeof emotionRecognizer != "undefined") {
      tf.engine().startScope();
      tf.tidy(() => {
        // Conversion to tensor4D and resize
        let tfImage = tf.browser.fromPixels(face, 3).expandDims(0);
        let tfResizedImage = tf.image.resizeBilinear(tfImage, [80, 80]);
        tfResizedImage = tfResizedImage.reshape([1, 80, 80, 3]);
        let prediction = Array.from(
          emotionRecognizer.predict(tfResizedImage).dataSync()
        );
        const currentEmotion = magnifyResults(emotions)(prediction);
        context.font = "48px serif";
        context.fillText(
          currentEmotion,
          xCenterBoundingBox * context.canvas.width,
          yCenterBoundingBox * context.canvas.height,
          widthBoundingBox * context.canvas.width
        );
        tfImage.dispose();
      });
      // Check tensor memory leak stop
      tf.engine().endScope();
    }
  }
};

const getPercentage = R.pipe(R.multiply(100), parseInt);

const getScoreInPercentage = R.map(getPercentage);

const getEmotionNearToItsScore = (listOfEmotions) => (pred) =>
  R.transpose([listOfEmotions, pred]);

const getListOfEmotionsSorted = R.sortBy(R.prop(1));

const magnifyResults = (listOfEmotions) =>
  R.pipe(
    getScoreInPercentage,
    getEmotionNearToItsScore(listOfEmotions),
    getListOfEmotionsSorted,
    R.reverse,
    R.nth(0),
    R.append(" %"),
    R.join("")
  );

const WebcamModified = () => {
  const { webcamRef, boundingBox, isLoading, detected, facesDetected } =
    useFaceDetection({
      faceDetectionOptions: {
        model: "short",
      },
      faceDetection: new FaceDetection.FaceDetection({
        locateFile: (file) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`,
      }),
      camera: ({ mediaSrc, onFrame, width, height }) =>
        new Camera(mediaSrc, {
          onFrame,
          width,
          height,
        }),
    });
  let canvasRef = useRef(null);

  const [model, setModel] = useState();
  const loadModel = async (link) => {
    try {
      if (typeof model === "undefined") {
        const fetchModel = await tf.loadLayersModel(link);
        setModel(fetchModel);
        console.log("load model success");
      }
    } catch (err) {
      console.log(err);
    }
  };
  tf.ready().then(() => loadModel(link));

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    let animationFrameId;
    const render = () => {
      drawOnCanvas(context, webcamRef.current.video, boundingBox, model);
      animationFrameId = window.requestAnimationFrame(render);
    };
    render();
    return () => window.cancelAnimationFrame(animationFrameId);
  });

  return (
    <div>
      <div
        style={{
          textAlign: "center",
          fontSize: "300%",
          position: "absolute",
          color: "black",
          backgroundColor: "white",
          borderBottomRightRadius: "20px",
          padding: "2%",
          opacity: 0.7,
        }}
      >
        {typeof model === "undefined" ? (
          <p>loading, please wait ...</p>
        ) : (
          <p>Emotion Recognition </p>
        )}
      </div>
      <div>
        <canvas
          ref={canvasRef}
          width={1920}
          height={1080}
          // style={cameraDisplayStyle}
          // style={{ display: "none" }}
        />
        <Webcam
          ref={webcamRef}
          width={1920}
          height={1080}
          style={{ display: "none" }}
        />
      </div>
    </div>
  );
};

export default WebcamModified;
